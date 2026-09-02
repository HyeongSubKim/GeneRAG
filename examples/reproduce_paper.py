# -*- coding: utf-8 -*-
"""
Reproduce the GeneRAG paper tables (Core HVG = Table 1, Global HVG = Table 2).

For one organ, this script loads the reference bank once, then for every
(backbone, anchor-gene-list) pair found in ``init_pred_fm_pt/`` it

1. scores the backbone's own linear-probe prediction on the anchor panel
   (the "GeneRAG ✗" rows of Table 1),
2. sweeps GeneRAG over the paper grid (α × ω, l1_ratio = 0.9), and for every
   configuration reports
   * Core-HVG PCC-10/50/K on the anchor panel        (Table 1 protocol),
   * Global-HVG PCC-k on the 10,000-gene bank panel   (Table 2 protocol),
   * the same Global metrics restricted to the top-5,000 HV genes.

Evaluation protocol (matches the paper's evaluation notebooks): ground truth
is log2(count + 1); GeneRAG output is calibrated with log1p; Pearson r is
computed per gene across the test-slide spots, then averaged over the top-k.

Run one organ per GPU, e.g.::

    python examples/reproduce_paper.py --organ PRAD   --device cuda:0
    python examples/reproduce_paper.py --organ kidney --device cuda:1
    python examples/reproduce_paper.py --organ her2st --device cuda:2

Results: ``hest1k_datasets/<organ>/results/reproduce_<organ>.csv``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from generag import GeneRAG
from generag.data import (
    load_bank_embeddings,
    load_bank_from_h5ad,
    load_gene_list,
    load_ground_truth,
    load_test_embeddings,
    load_test_predictions,
)
from generag.metrics import evaluate_predictions, gene_pearson_array

# Paper splits: held-out test slide + bank slides, and the Core-HVG panel size K.
ORGANS = {
    "PRAD":   dict(test="MEND145", train=[f"MEND{i}" for i in range(139, 163) if i not in (145, 155)], core_k=200),
    "kidney": dict(test="NCBI697", train=[f"NCBI{i}" for i in range(692, 715) if i != 697], core_k=200),
    "her2st": dict(test="SPA148",  train=[f"SPA{i}" for i in range(119, 154) if i != 148], core_k=300),
}
# Backbone -> embedding directory / file suffix (notebook 02 convention).
EMBEDDINGS = {
    "UNI":    ("1spot_uni_ebd_aug", "_uni_aug.pt"),
    "Exaone": ("1spot_exaone_ebd_aug", "_exaone_aug.pt"),
    "CONCH":  ("1spot_conch_ebd_aug", "_conch_aug.pt"),
}
GRID = dict(alpha=[0.001, 0.01, 0.1], l1_ratio=[0.9], embedding_ratio=[0.0, 0.25, 0.5, 0.75, 1.0])
TOP_KS = (10, 50, 200, 300, 1000, 2000, 3000, 5000, 10000)
N_HV = 10_000


def core_metrics(pred: pd.DataFrame, gt_log2: pd.DataFrame, genes: list[str], k: int, prefix: str) -> dict:
    """PCC-10/50/K over the anchor panel only (Table 1 protocol)."""
    r, _ = gene_pearson_array(pred[genes], gt_log2, genes)
    r = np.sort(r[np.isfinite(r)])[::-1]
    return {
        f"{prefix}pcc_10": float(r[:10].mean()),
        f"{prefix}pcc_50": float(r[:50].mean()),
        f"{prefix}pcc_{k}": float(r[:k].mean()),
        f"{prefix}n_genes": int(r.size),
    }


def discover(pred_dir: str, gene_dir: str) -> list[tuple[str, str, str]]:
    """(backbone, gene_basename, pred_path) for every matching prediction file."""
    out = []
    for f in sorted(glob.glob(os.path.join(pred_dir, "generated_samples_lr_*_*_20sample.pt"))):
        inner = os.path.basename(f)[len("generated_samples_lr_"):-len("_20sample.pt")]
        bb, _, gene = inner.partition("_")
        if gene and os.path.isfile(os.path.join(gene_dir, f"selected_{gene}_list.txt")):
            out.append((bb, gene, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", required=True, choices=list(ORGANS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", default="./hest1k_datasets")
    ap.add_argument("--only-models", default="", help="comma list, e.g. UNI,Exaone")
    ap.add_argument("--only-genes", default="", help="comma list of gene basenames")
    ap.add_argument("--tag", default="", help="suffix for the output CSV (to split one organ across GPUs)")
    args = ap.parse_args()

    cfg = ORGANS[args.organ]
    root = os.path.join(args.data_root, args.organ)
    st_path, proc, pred_dir = (os.path.join(root, d) for d in ("st", "processed_data", "init_pred_fm_pt"))
    out_dir = os.path.join(root, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"reproduce_{args.organ}{'_' + args.tag if args.tag else ''}.csv")

    tasks = discover(pred_dir, proc)
    if args.only_models:
        tasks = [t for t in tasks if t[0] in args.only_models.split(",")]
    if args.only_genes:
        tasks = [t for t in tasks if t[1] in args.only_genes.split(",")]
    print(f"[{args.organ}] test={cfg['test']} bank={len(cfg['train'])} slides | tasks={[(b, g) for b, g, _ in tasks]}", flush=True)

    # Bank: full gene panel, restricted to genes present in every bank slide.
    t0 = time.time()
    _, bank_all = load_bank_from_h5ad(cfg["train"], [], st_path)
    bank_all = bank_all.dropna(axis=1)
    print(f"[{args.organ}] bank {bank_all.shape} loaded in {time.time() - t0:.0f}s", flush=True)

    rows: list[dict] = []
    emb_cache: dict[str, np.ndarray | None] = {}
    for bb, gene, pred_path in tasks:
        gene_list = load_gene_list(os.path.join(proc, f"selected_{gene}_list.txt"))
        test_anchor, spots = load_test_predictions(pred_path, cfg["test"], gene_list, st_path)
        gt = load_ground_truth(spots, cfg["test"], st_path, log2=True)
        anchors = [g for g in test_anchor.columns if g in bank_all.columns and g in gt.columns]
        k = cfg["core_k"]

        base = dict(organ=args.organ, backbone=bb, gene_list=gene, n_anchor=len(anchors))
        # Table 1 "✗" row: the backbone's own prediction on the anchor panel.
        rows.append({**base, "config": "init", **core_metrics(test_anchor, gt, anchors, k, "core_")})
        print(f"  {bb}/{gene} init: {rows[-1]}", flush=True)

        # Embeddings (bank + test) for this backbone.
        if bb not in emb_cache:
            sub, suffix = EMBEDDINGS.get(bb, (None, None))
            edir = os.path.join(proc, sub) if sub else None
            if edir and os.path.isdir(edir):
                emb_cache[bb] = (
                    load_bank_embeddings(edir, bank_all.index.tolist(), suffix),
                    load_test_embeddings(edir, cfg["test"], spots, suffix),
                )
            else:
                emb_cache[bb] = (None, None)
        bank_emb, test_emb = emb_cache[bb]
        if bank_emb is None:
            print(f"  {bb}: no embeddings found -> embedding_ratio>0 configs skipped", flush=True)

        model = GeneRAG(bank_all, bank_embeddings=bank_emb, anchor_genes=anchors, n_high_variable_genes=N_HV)
        print(f"  {model}", flush=True)

        for alpha, l1, omega in product(GRID["alpha"], GRID["l1_ratio"], GRID["embedding_ratio"]):
            if omega > 0 and bank_emb is None:
                continue
            t0 = time.time()
            pred, sparsity = model.predict(
                test_anchor, test_emb, method="elasticnet", alpha=alpha, l1_ratio=l1,
                embedding_ratio=omega, positive=True, device=args.device,
            )
            row = {**base, "config": "generag", "alpha": alpha, "l1_ratio": l1, "omega": omega, "sparsity": sparsity}
            # Anchors that survived the HV-panel filter inside GeneRAG.
            row.update(core_metrics(np.log1p(pred[model.anchor_genes]), gt, model.anchor_genes, k, "core_"))
            row.update({f"hv10k_{m}": v for m, v in evaluate_predictions(pred, gt, "log1p", TOP_KS).items()})
            row.update({f"hv5k_{m}": v for m, v in evaluate_predictions(pred.iloc[:, :5000], gt, "log1p", TOP_KS[:-1]).items()})
            row["seconds"] = time.time() - t0
            rows.append(row)
            print(f"  {bb}/{gene} a={alpha} w={omega}: nnz={sparsity:.1f} core10={row['core_pcc_10']:.4f} "
                  f"core{k}={row[f'core_pcc_{k}']:.4f} hv10k_pcc10={row['hv10k_pcc_10']:.4f} "
                  f"hv10k_pcc300={row['hv10k_pcc_300']:.4f} hv5k_pcc5000={row['hv5k_pcc_5000']:.4f} ({row['seconds']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[{args.organ}] wrote {out_csv} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
