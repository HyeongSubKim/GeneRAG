# -*- coding: utf-8 -*-
"""
End-to-end paper-style experiment driver.

Reproduces the GeneRAG sweep used in the paper: for each saved
``generated_samples_lr_{MODEL}_{gene_basename}_20sample.pt`` file, build
the per-tissue reference bank, run a hyperparameter sweep, and write a
CSV of per-experiment metrics.

This is the new replacement for the old ``main.py`` and uses the
:func:`generag.experiment.run_sweep` API.

Edit the configuration block at the top and run:

    python examples/run_experiment.py
"""

from __future__ import annotations

import glob
import os
import sys

# Make ``generag`` importable when running the example from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generag.data import (
    load_bank_from_h5ad,
    load_bank_embeddings,
    load_gene_list,
    load_ground_truth,
    load_test_embeddings,
    load_test_predictions,
)
from generag.experiment import run_sweep


# =============================================================================
# Configuration
# =============================================================================

EXP_NAME = "GeneRAG_PFM_test_top200"
ORGAN = "PRAD"

# All paths are organ-scoped under ./hest1k_datasets/<ORGAN>/ — the same directory
# layout produced by examples/notebooks/01_download_data.ipynb → 02 → 03.
DATA_ROOT = "./hest1k_datasets"
DATA_PATH = f"{DATA_ROOT}/{ORGAN}/processed_data"
ST_PATH = f"{DATA_ROOT}/{ORGAN}/st"
LR_PRED_PT_DIR = f"{DATA_ROOT}/{ORGAN}/init_pred_fm_pt"
SELECTED_GENE_DIR = f"{DATA_ROOT}/{ORGAN}/processed_data"
SAVE_DIR = f"{DATA_ROOT}/{ORGAN}/results"

TEST_SLIDE = "MEND145"
TRAIN_SLIDES = [s for s in (f"MEND{i}" for i in range(139, 163)) if s not in {"MEND155", TEST_SLIDE}]

N_HIGH_VAR_GENES = 10_000
CALIBRATION = "log1p"
N_JOBS = 1  # set >1 to round-robin across GPUs via ProcessPoolExecutor

# (method, params grid) -> sweep. The default grid below reproduces the paper's
# best-of-sweep config (α=0.1, l1_ratio=0.9, ω=1.0) for UNI on Prostate; widen
# the grid to explore other operating points.
SEARCH_SPACE = {
    "elasticnet": {
        "alpha": [0.001, 0.01, 0.1],
        "l1_ratio": [0.9],
        "embedding_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        "positive": [True],
        "max_iter": [2000],
    },
}


# =============================================================================
# Discovery: pair LR-prediction .pt files with their gene list
# =============================================================================

def discover_tasks(pred_dir: str, gene_dir: str) -> list[tuple[str, str, str, str]]:
    """Yield ``(model_name, pred_path, gene_basename, gene_list_filename)``.

    Naming convention used by the released checkpoints::

        generated_samples_lr_{MODEL}_{gene_basename}_20sample.pt
        selected_{gene_basename}_list.txt
    """
    tasks: list[tuple[str, str, str, str]] = []
    for fpath in sorted(glob.glob(os.path.join(pred_dir, "generated_samples_lr_*_*_20sample.pt"))):
        basename = os.path.basename(fpath)
        inner = basename[len("generated_samples_lr_"):-len("_20sample.pt")]
        model_name, _, gene_basename = inner.partition("_")
        if not gene_basename:
            continue
        gene_list_file = f"selected_{gene_basename}_list.txt"
        if os.path.isfile(os.path.join(gene_dir, gene_list_file)):
            tasks.append((model_name, fpath, gene_basename, gene_list_file))
    return tasks


# =============================================================================
# Main loop
# =============================================================================

def main():
    tasks = discover_tasks(LR_PRED_PT_DIR, SELECTED_GENE_DIR)
    if not tasks:
        raise SystemExit(f"No prediction .pt files matched in {LR_PRED_PT_DIR}")

    # Optional filters (set to None to include everything).
    only_models = os.environ.get("ONLY_MODELS")   # e.g. "UNI" or "UNI,Exaone"
    only_genes = os.environ.get("ONLY_GENES")     # e.g. "morph200_8"
    if only_models:
        keep = set(only_models.split(","))
        tasks = [t for t in tasks if t[0] in keep]
    if only_genes:
        keep = set(only_genes.split(","))
        tasks = [t for t in tasks if t[2] in keep]
    if not tasks:
        raise SystemExit("No tasks left after ONLY_MODELS / ONLY_GENES filters.")

    print(f"Test slide: {TEST_SLIDE}  |  tasks: {len(tasks)}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    for model_name, pred_path, gene_basename, gene_list_file in tasks:
        print(f"\n=== {model_name} / {gene_basename} ===")

        anchor_genes = load_gene_list(os.path.join(SELECTED_GENE_DIR, gene_list_file))
        bank_selected_df, bank_all_df = load_bank_from_h5ad(TRAIN_SLIDES, anchor_genes, ST_PATH)
        test_anchor, test_spots = load_test_predictions(pred_path, TEST_SLIDE, anchor_genes, ST_PATH)
        test_gt = load_ground_truth(test_spots, TEST_SLIDE, ST_PATH)

        # Optional embeddings (only needed if embedding_ratio > 0).
        ebd_subdir = "1spot_exaone_ebd_aug" if model_name == "Exaone" else "1spot_uni_ebd_aug"
        embed_dir = os.path.join(DATA_PATH, ebd_subdir)
        bank_emb = load_bank_embeddings(embed_dir, bank_all_df.index.tolist()) if os.path.isdir(embed_dir) else None
        test_emb = load_test_embeddings(embed_dir, TEST_SLIDE, test_spots) if os.path.isdir(embed_dir) else None

        out_csv = os.path.join(
            SAVE_DIR,
            f"{EXP_NAME}_{model_name}_{os.path.splitext(gene_list_file)[0]}.csv",
        )
        run_sweep(
            bank_expression=bank_all_df,
            test_anchor=test_anchor,
            test_gt=test_gt,
            anchor_genes=anchor_genes,
            bank_embeddings=bank_emb,
            test_embeddings=test_emb,
            search_space=SEARCH_SPACE,
            n_high_var_genes=N_HIGH_VAR_GENES,
            calibration=CALIBRATION,
            n_jobs=N_JOBS,
            output_csv=out_csv,
        )


if __name__ == "__main__":
    main()
