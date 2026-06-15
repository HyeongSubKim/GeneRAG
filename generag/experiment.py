# -*- coding: utf-8 -*-
"""
Hyperparameter sweep runner around :class:`generag.GeneRAG`.

This module turns "I want to evaluate GeneRAG over a grid of method ×
hyperparameters" into a single call. It builds the bank once, then runs
every (method, params) combination, collects evaluation metrics, and can
distribute the combinations across multiple GPUs.

For one-off prediction use :class:`generag.GeneRAG` directly; this module
exists for paper-style hyperparameter studies.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import torch

from .core import GeneRAG
from .metrics import evaluate_predictions


# =============================================================================
# Default search space
# =============================================================================

# Light defaults that match the paper's reported configuration; users can
# pass their own search_space dict.
DEFAULT_SEARCH_SPACE: dict[str, dict[str, list]] = {
    "elasticnet": {
        "alpha": [0.01],
        "l1_ratio": [0.9],
        "embedding_ratio": [0.75],
    },
}


# =============================================================================
# Search-space expansion
# =============================================================================

def _expand_search_space(search_space: dict[str, dict[str, list]]) -> list[tuple[str, dict[str, Any]]]:
    """Cartesian-product expansion of ``method -> {param: [values]}``."""
    tasks: list[tuple[str, dict[str, Any]]] = []
    for method, grid in search_space.items():
        if not grid:
            tasks.append((method, {}))
            continue
        keys = list(grid.keys())
        for combo in product(*[grid[k] for k in keys]):
            tasks.append((method, dict(zip(keys, combo))))
    return tasks


# =============================================================================
# Single-experiment worker (also runs in subprocess for multi-GPU mode)
# =============================================================================

def _run_one(args):
    """Build a fresh GeneRAG instance and evaluate one (method, params) combination.

    Re-building the model inside each worker keeps multi-GPU execution simple
    (no shared in-memory state, no pickling of CUDA tensors).
    """
    (
        exp_id,
        method,
        params,
        bank_expression,
        bank_embeddings,
        anchor_genes,
        test_anchor,
        test_embeddings,
        test_gt,
        n_high_var_genes,
        calibration,
        device_id,
    ) = args

    device = (
        f"cuda:{device_id}"
        if device_id is not None and torch.cuda.is_available() and device_id < torch.cuda.device_count()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    embedding_ratio = float(np.clip(params.get("embedding_ratio", 0.0), 0.0, 1.0))
    solver_params = {k: v for k, v in params.items() if k != "embedding_ratio"}

    try:
        model = GeneRAG(
            bank_expression=bank_expression,
            bank_embeddings=bank_embeddings,
            anchor_genes=anchor_genes,
            n_high_variable_genes=n_high_var_genes,
        )
        predicted, sparsity = model.predict(
            test_anchor=test_anchor,
            test_embeddings=test_embeddings,
            method=method,
            embedding_ratio=embedding_ratio,
            device=device,
            **{k: v for k, v in solver_params.items() if k in ("alpha", "l1_ratio", "positive", "max_iter", "tol")},
        )
        metrics = evaluate_predictions(predicted, test_gt, calibration=calibration)
        return {
            "experiment_id": exp_id,
            "optimization_method": method,
            "sparsity": sparsity,
            **params,
            "embedding_ratio": embedding_ratio,
            **metrics,
        }
    except Exception as exc:  # pragma: no cover — surfacing solver failures
        return {
            "experiment_id": exp_id,
            "optimization_method": method,
            "sparsity": np.nan,
            **params,
            "embedding_ratio": embedding_ratio,
            "error": repr(exc),
        }


# =============================================================================
# Public runner
# =============================================================================

def run_sweep(
    *,
    bank_expression: pd.DataFrame,
    test_anchor: pd.DataFrame,
    test_gt: pd.DataFrame,
    anchor_genes: list[str] | None = None,
    bank_embeddings: np.ndarray | None = None,
    test_embeddings: np.ndarray | None = None,
    search_space: dict[str, dict[str, list]] | None = None,
    n_high_var_genes: int = 10_000,
    calibration: str | None = "log1p",
    n_jobs: int = 1,
    output_csv: str | None = None,
    save_intermediate: bool = True,
) -> pd.DataFrame:
    """Run a (method × hyperparameter) sweep around GeneRAG.

    Parameters
    ----------
    bank_expression, test_anchor, test_gt
        Same inputs as :class:`GeneRAG` plus a ground-truth dataframe used
        for evaluation.
    anchor_genes, bank_embeddings, test_embeddings, n_high_var_genes
        Forwarded to :class:`GeneRAG`.
    search_space
        ``{method: {param_name: [values, ...]}}``. Defaults to
        :data:`DEFAULT_SEARCH_SPACE`.
    calibration
        Forwarded to :func:`generag.metrics.evaluate_predictions`.
    n_jobs
        ``> 1`` enables a ``ProcessPoolExecutor`` and round-robins tasks
        across available CUDA devices.
    output_csv
        Optional path to write the final results to (also used to derive
        an ``_intermediate`` file written periodically when
        ``save_intermediate=True``).
    """
    search_space = search_space or DEFAULT_SEARCH_SPACE
    tasks = _expand_search_space(search_space)
    n_total = len(tasks)
    n_gpus = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1

    print(f"Sweep: {n_total} experiments | methods={list(search_space)} | n_jobs={n_jobs} | GPUs={n_gpus}")
    started = datetime.now()

    results: list[dict] = []

    if n_jobs > 1:
        # Multi-GPU: round-robin device assignment, one process per worker.
        workers = min(n_jobs, n_gpus)
        args_iter = [
            (
                i + 1, method, params,
                bank_expression, bank_embeddings, anchor_genes,
                test_anchor, test_embeddings, test_gt,
                n_high_var_genes, calibration,
                i % n_gpus,
            )
            for i, (method, params) in enumerate(tasks)
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, a): a[0] for a in args_iter}
            for fut in as_completed(futures):
                results.append(fut.result())
                _maybe_save_intermediate(results, output_csv, save_intermediate, n_total)
    else:
        # Single GPU (or CPU) sequential execution.
        for i, (method, params) in enumerate(tasks):
            args = (
                i + 1, method, params,
                bank_expression, bank_embeddings, anchor_genes,
                test_anchor, test_embeddings, test_gt,
                n_high_var_genes, calibration,
                0 if torch.cuda.is_available() else None,
            )
            results.append(_run_one(args))
            _maybe_save_intermediate(results, output_csv, save_intermediate, n_total)

    df = pd.DataFrame(results)
    if "experiment_id" in df.columns:
        df = df.sort_values("experiment_id").drop(columns=["experiment_id"])
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Wrote final results to {output_csv}")

    print(f"Total time: {datetime.now() - started}  ({len(df)} experiments)")
    return df


def _maybe_save_intermediate(
    results: list[dict],
    output_csv: str | None,
    save_intermediate: bool,
    n_total: int,
):
    """Periodically dump an ``_intermediate`` CSV so long sweeps survive crashes."""
    if not (save_intermediate and output_csv):
        return
    every = max(1, n_total // 10 + 1)
    if len(results) % every != 0:
        return
    base, ext = os.path.splitext(output_csv)
    path = f"{base}_intermediate{ext}"
    df = pd.DataFrame(results)
    if "experiment_id" in df.columns:
        df = df.sort_values("experiment_id")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
