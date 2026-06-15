# -*- coding: utf-8 -*-
"""
Evaluation metrics for spatial-transcriptomics reconstruction.

Conventions
-----------
* ``predicted`` and ``ground_truth`` are both ``pandas.DataFrame``s
  indexed by spot, with gene-name columns.
* Pearson correlation is computed per-gene across spots, mirroring the
  metric used in HEST-1k and related benchmarks.
* Top-K Pearson means follow the GeneRAG paper: genes are ranked by
  per-gene Pearson r, then averaged over the top K.
* MSE / MAE / RVD are computed on the top-300 most predictable genes
  (selected by Pearson r) to align with the GeneRAG paper.

The :func:`evaluate_predictions` function is the high-level entry point;
the lower-level helpers are exposed for ad-hoc diagnostics.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Calibration
# =============================================================================

def calibrate(df: pd.DataFrame, method: str | None = None) -> pd.DataFrame:
    """Apply a column-wise calibration to the predicted expression matrix.

    Supported methods
    -----------------
    ``'log1p'`` : ``log(1 + x)``    (default for raw-count style outputs)
    ``'log2'``  : ``log2(1 + x)``
    ``'zscore'``: per-column standardisation
    ``'quantile'``: column-wise rank-based Gaussianisation
    ``None``    : identity
    """
    if method is None:
        return df.copy()
    if method == "log1p":
        return np.log1p(df)
    if method == "log2":
        return np.log2(df + 1)
    if method == "zscore":
        out = df.copy()
        for col in df.columns:
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                out[col] = (df[col] - mean) / std
        return out
    if method == "quantile":
        out = df.copy()
        for col in df.columns:
            if df[col].notna().sum() > 0:
                ranks = stats.rankdata(df[col].fillna(0), method="average")
                out[col] = stats.norm.ppf(ranks / (len(df) + 1))
        return out
    raise ValueError(f"Unknown calibration method: {method!r}")


# =============================================================================
# Per-gene / per-spot Pearson r
# =============================================================================

def gene_pearson(predicted: pd.DataFrame, ground_truth: pd.DataFrame, gene: str) -> float:
    """Pearson correlation between ``predicted[gene]`` and ``ground_truth[gene]``."""
    if gene not in predicted.columns or gene not in ground_truth.columns:
        return np.nan
    common = list(set(predicted.index) & set(ground_truth.index))
    a = predicted.loc[common, gene].to_numpy()
    b = ground_truth.loc[common, gene].to_numpy()
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def spot_pearson(predicted: pd.DataFrame, ground_truth: pd.DataFrame, spot: str) -> float:
    """Pearson correlation between predicted and ground-truth profiles at ``spot``."""
    if spot not in predicted.index or spot not in ground_truth.index:
        return np.nan
    common = list(set(predicted.columns) & set(ground_truth.columns))
    a = predicted.loc[spot, common].to_numpy()
    b = ground_truth.loc[spot, common].to_numpy()
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def gene_pearson_array(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
    genes: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Per-gene Pearson r for many genes at once.

    Returns
    -------
    correlations : ndarray
        Pearson r per gene (NaN entries removed in caller as needed).
    gene_order : list of str
        Genes used, in the same order as ``correlations``.
    """
    if genes is None:
        genes = list(predicted.columns)
    common_spots = list(set(predicted.index) & set(ground_truth.index))
    if not common_spots:
        return np.array([]), []

    P = predicted.loc[common_spots, [g for g in genes if g in predicted.columns]].to_numpy()
    GT_cols = [g for g in genes if g in predicted.columns and g in ground_truth.columns]
    if not GT_cols:
        return np.array([]), []
    P = predicted.loc[common_spots, GT_cols].to_numpy()
    G = ground_truth.loc[common_spots, GT_cols].to_numpy()

    # Vectorised Pearson r per column.
    Pm = P - np.nanmean(P, axis=0, keepdims=True)
    Gm = G - np.nanmean(G, axis=0, keepdims=True)
    num = np.nansum(Pm * Gm, axis=0)
    den = np.sqrt(np.nansum(Pm**2, axis=0) * np.nansum(Gm**2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return r, GT_cols


# =============================================================================
# High-level evaluation
# =============================================================================

# Default Top-K cut-offs reported by the GeneRAG paper.
DEFAULT_TOP_KS = (10, 50, 300, 1000, 2000, 3000, 5000, 10000)


def evaluate_predictions(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
    calibration: str | None = "log1p",
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
    error_top_k: int = 300,
) -> dict[str, float]:
    """Compute the full evaluation panel (PCC@K, MSE, MAE, RVD).

    Steps
    -----
    1. Calibrate ``predicted`` (e.g. ``log1p``).
    2. Compute per-gene Pearson r across spots.
    3. Sort genes by Pearson r and report Top-K means for ``top_ks``.
    4. Compute MSE / MAE / RVD over the top ``error_top_k`` genes.

    Returns
    -------
    metrics : dict
        Keys: ``pcc_10`` ... (per ``top_ks``), ``mse``, ``mae``, ``rvd``.
    """
    P = calibrate(predicted, calibration)

    # 1. Per-gene Pearson r.
    r, gene_order = gene_pearson_array(P, ground_truth)
    if r.size == 0:
        empty = {f"pcc_{k}": np.nan for k in top_ks}
        empty.update({"mse": np.nan, "mae": np.nan, "rvd": np.nan})
        return empty

    # 2. Top-K averages over finite-only Pearson r (NaN genes — zero variance,
    # missing values, etc. — are dropped before ranking, matching the
    # convention used in the GeneRAG paper and HEST-1k benchmark).
    valid_r = r[np.isfinite(r)]
    sorted_r = np.sort(valid_r)[::-1] if valid_r.size else np.array([])
    out: dict[str, float] = {}
    for k in top_ks:
        out[f"pcc_{k}"] = float(np.mean(sorted_r[:k])) if sorted_r.size >= k else (
            float(np.mean(sorted_r)) if sorted_r.size else np.nan
        )

    # 3. MSE / MAE / RVD on the top error_top_k genes.
    order = np.argsort(-np.nan_to_num(r, nan=-np.inf))
    top_genes = [gene_order[i] for i in order[:error_top_k]]
    common_spots = list(set(P.index) & set(ground_truth.index))
    A = P.loc[common_spots, top_genes].to_numpy()
    B = ground_truth.loc[common_spots, top_genes].to_numpy()
    mask = ~(np.isnan(A) | np.isnan(B))
    if mask.any():
        diff = (B - A)[mask]
        out["mse"] = float(np.mean(diff**2))
        out["mae"] = float(np.mean(np.abs(diff)))
    else:
        out["mse"] = np.nan
        out["mae"] = np.nan

    # RVD: ((var_pred - var_gt) / var_gt) ^ 2 averaged over genes with non-zero gt variance.
    var_pred = np.nanvar(A, axis=0)
    var_gt = np.nanvar(B, axis=0)
    valid = np.isfinite(var_pred) & np.isfinite(var_gt) & (var_gt > 0)
    if valid.any():
        rvd_values = ((var_pred[valid] - var_gt[valid]) / var_gt[valid]) ** 2
        out["rvd"] = float(np.mean(rvd_values))
    else:
        out["rvd"] = np.nan

    return out
