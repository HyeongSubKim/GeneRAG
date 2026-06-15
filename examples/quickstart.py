# -*- coding: utf-8 -*-
"""
Quickstart — the plug-and-play GeneRAG API in ~30 lines.

This script shows the smallest end-to-end use of GeneRAG: build a model
from any in-memory reference bank, then predict gene expression for new
spots. It uses synthetic data so it runs anywhere without dependencies on
specific datasets.

Run with:

    python examples/quickstart.py
"""

from __future__ import annotations

import os
import sys

# Make ``generag`` importable when running the example from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from generag import GeneRAG
from generag.utils import auto_device, set_global_seed


def main():
    set_global_seed(0)

    # ---- Synthetic reference bank --------------------------------------
    n_bank, n_genes, emb_dim = 2_000, 1_500, 64
    rng = np.random.default_rng(0)
    bank_expr = pd.DataFrame(
        rng.gamma(shape=2.0, scale=1.0, size=(n_bank, n_genes)),
        index=[f"bank_spot_{i}" for i in range(n_bank)],
        columns=[f"GENE{i}" for i in range(n_genes)],
    )
    bank_emb = rng.standard_normal((n_bank, emb_dim))

    # ---- Anchor gene subset (e.g. the panel you can already predict) ---
    anchor_genes = bank_expr.columns[:200].tolist()

    # ---- Test inputs ---------------------------------------------------
    n_test = 50
    test_anchor = pd.DataFrame(
        rng.gamma(shape=2.0, scale=1.0, size=(n_test, len(anchor_genes))),
        index=[f"test_spot_{i}" for i in range(n_test)],
        columns=anchor_genes,
    )
    test_emb = rng.standard_normal((n_test, emb_dim))

    # ---- Build GeneRAG and predict -------------------------------------
    model = GeneRAG(
        bank_expression=bank_expr,
        bank_embeddings=bank_emb,
        anchor_genes=anchor_genes,
        n_high_variable_genes=1_000,
    )
    print(model)

    predictions, mean_sparsity = model.predict(
        test_anchor=test_anchor,
        test_embeddings=test_emb,
        method="elasticnet",
        alpha=0.01,
        l1_ratio=0.9,
        embedding_ratio=0.75,
        positive=True,
        device=auto_device("cuda"),
    )
    print(f"predictions shape: {predictions.shape} (test_spots × HV-genes)")
    print(f"mean sparsity per spot: {mean_sparsity:.2f}")
    print(predictions.iloc[:3, :5])


if __name__ == "__main__":
    main()
