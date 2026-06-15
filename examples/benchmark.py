# -*- coding: utf-8 -*-
"""
Benchmark — measure slide-wise and spot-wise inference time on one task.

Mirrors the paper's reported timing methodology. Single CUDA device,
single (model × gene-list) combination, full reference bank. Useful as a
sanity check that your install is hitting the GPU solver.

Edit the configuration block and run:

    python examples/benchmark.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generag import GeneRAG
from generag.data import (
    load_bank_embeddings,
    load_bank_from_h5ad,
    load_gene_list,
    load_ground_truth,
    load_test_embeddings,
    load_test_predictions,
)
from generag.utils import auto_device


# =============================================================================
# Configuration
# =============================================================================

ORGAN = "PRAD"

# Organ-scoped paths — same layout produced by examples/notebooks/01_download_data.ipynb → 02 → 03.
DATA_ROOT = "./hest1k_datasets"
DATA_PATH = f"{DATA_ROOT}/{ORGAN}/processed_data"
ST_PATH = f"{DATA_ROOT}/{ORGAN}/st"
LR_PRED_PT_DIR = f"{DATA_ROOT}/{ORGAN}/init_pred_fm_pt"
SELECTED_GENE_DIR = f"{DATA_ROOT}/{ORGAN}/processed_data"

TEST_SLIDE = "MEND145"
TRAIN_SLIDES = [s for s in (f"MEND{i}" for i in range(139, 163)) if s not in {"MEND155", TEST_SLIDE}]

MODEL_NAME = "UNI"
GENE_BASENAME = "morph200_8"
PRED_PATH = os.path.join(LR_PRED_PT_DIR, f"generated_samples_lr_{MODEL_NAME}_{GENE_BASENAME}_20sample.pt")
GENE_LIST_FILE = os.path.join(SELECTED_GENE_DIR, f"selected_{GENE_BASENAME}_list.txt")
EMBED_DIR = os.path.join(DATA_PATH, "1spot_uni_ebd_aug" if MODEL_NAME != "Exaone" else "1spot_exaone_ebd_aug")

METHOD = "elasticnet"
ALPHA = 0.01
L1_RATIO = 0.9
EMBEDDING_RATIO = 0.75
POSITIVE = True
MAX_ITER = 2000
N_HIGH_VAR_GENES = 10_000


def main():
    device = auto_device("cuda:0")
    print(f"Device: {device}  |  method: {METHOD}  |  embedding_ratio: {EMBEDDING_RATIO}")

    # Warm up CUDA so timing reflects steady-state.
    if device.startswith("cuda"):
        _ = torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
        torch.cuda.synchronize()

    # ---- Load data (excluded from timing) ------------------------------
    print("--- Loading ---")
    anchor_genes = load_gene_list(GENE_LIST_FILE)
    bank_selected_df, bank_all_df = load_bank_from_h5ad(TRAIN_SLIDES, anchor_genes, ST_PATH)
    test_anchor, test_spots = load_test_predictions(PRED_PATH, TEST_SLIDE, anchor_genes, ST_PATH)
    _ = load_ground_truth(test_spots, TEST_SLIDE, ST_PATH)
    bank_emb = load_bank_embeddings(EMBED_DIR, bank_all_df.index.tolist())
    test_emb = load_test_embeddings(EMBED_DIR, TEST_SLIDE, test_spots)

    n_spots = len(test_spots)
    n_bank = len(bank_all_df)
    print(f"  test slide: {TEST_SLIDE}  |  n_spots={n_spots}  |  bank_spots={n_bank}")

    # ---- Build model (also excluded — one-time cost) -------------------
    model = GeneRAG(
        bank_expression=bank_all_df,
        bank_embeddings=bank_emb,
        anchor_genes=anchor_genes,
        n_high_variable_genes=N_HIGH_VAR_GENES,
    )

    # ---- Time the inference --------------------------------------------
    print("--- GeneRAG inference ---")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model.predict(
        test_anchor=test_anchor,
        test_embeddings=test_emb,
        method=METHOD,
        alpha=ALPHA,
        l1_ratio=L1_RATIO,
        embedding_ratio=EMBEDDING_RATIO,
        positive=POSITIVE,
        max_iter=MAX_ITER,
        device=device,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    print(f"\n{'='*60}\nResult (test={TEST_SLIDE}, n_spots={n_spots}, n_bank={n_bank})\n{'='*60}")
    print(f"  slide-wise total : {wall:8.2f} s   ({wall/60:.2f} min)")
    print(f"  spot-wise avg    : {wall / n_spots * 1000:8.2f} ms/spot")


if __name__ == "__main__":
    main()
