# -*- coding: utf-8 -*-
"""
Compare ``reproduce_paper.py`` outputs against the numbers reported in the
GeneRAG paper (Table 1: Core HVG, Table 2: Global HVG, 5,000 genes).

    python examples/compare_to_paper.py            # all organs found under hest1k_datasets/

For every (organ, backbone) it prints
* the backbone-only baseline (paper "✗" row) next to our linear-probe score,
* GeneRAG at the fixed default config (α=0.01, ω=0.75) and at the best
  configuration of the sweep, next to the paper "✓" row,
* the Global-HVG PCC-k curve next to Table 2.
For Prostate the three ``morph200_{7,8,9}`` anchor lists are averaged.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

# --- Numbers from the paper / project page --------------------------------
# Table 1 (Core HVG): PCC-10, PCC-50, PCC-K  (K = 300 breast, 200 kidney/prostate)
TABLE1 = {
    ("her2st", "UNI", False):    (0.8301, 0.7909, 0.6024),
    ("her2st", "UNI", True):     (0.8670, 0.8257, 0.7017),
    ("her2st", "Exaone", False): (0.8217, 0.7850, 0.6251),
    ("her2st", "Exaone", True):  (0.8589, 0.8175, 0.7002),
    ("her2st", "CONCH", False):  (0.7799, 0.7467, 0.6043),
    ("kidney", "UNI", False):    (0.4828, 0.3888, 0.2715),
    ("kidney", "UNI", True):     (0.5529, 0.4987, 0.3525),
    ("kidney", "Exaone", False): (0.4584, 0.4023, 0.3009),
    ("kidney", "Exaone", True):  (0.5479, 0.4886, 0.3347),
    ("kidney", "CONCH", False):  (0.3583, 0.3109, 0.2243),
    ("PRAD", "UNI", False):      (0.5548, 0.4761, 0.3076),
    ("PRAD", "UNI", True):       (0.6801, 0.6322, 0.5046),
    ("PRAD", "Exaone", False):   (0.6204, 0.5313, 0.3536),
    ("PRAD", "Exaone", True):    (0.6911, 0.6513, 0.5371),
    ("PRAD", "CONCH", False):    (0.5660, 0.4715, 0.3171),
}
# Table 2 (Global HVG): PCC-10/50/300/1000/2000/3000/5000
TABLE2 = {
    ("her2st", "UNI"):    (0.8716, 0.8393, 0.7894, 0.7297, 0.6721, 0.6248, 0.5465),
    ("her2st", "Exaone"): (0.8619, 0.8316, 0.7825, 0.7254, 0.6728, 0.6301, 0.5591),
    ("kidney", "UNI"):    (0.5666, 0.5182, 0.4189, 0.3148, 0.2558, 0.2236, 0.1828),
    ("kidney", "Exaone"): (0.5601, 0.5069, 0.4084, 0.3107, 0.2495, 0.2149, 0.1715),
    ("PRAD", "UNI"):      (0.6840, 0.6588, 0.5875, 0.4902, 0.4157, 0.3663, 0.1980),
    ("PRAD", "Exaone"):   (0.7051, 0.6800, 0.6042, 0.4977, 0.4187, 0.3669, 0.2981),
}
T2_KS = (10, 50, 300, 1000, 2000, 3000, 5000)
CORE_K = {"her2st": 300, "kidney": 200, "PRAD": 200}
DEFAULT_CFG = dict(alpha=0.01, omega=0.75)
# Anchor lists that correspond to the paper's Core-HVG panel, per organ.
PAPER_LISTS = {"PRAD": ["morph200_7", "morph200_8", "morph200_9"], "kidney": ["morph200"], "her2st": ["top300"]}
# Override which anchor lists are pooled per organ, e.g. PRAD_LISTS=gene (notebook-02 list) or PRAD_LISTS=top200.
if os.environ.get("PRAD_LISTS"):
    PAPER_LISTS["PRAD"] = os.environ["PRAD_LISTS"].split(",")


def fmt(vals) -> str:
    return " ".join(f"{v:.4f}" if np.isfinite(v) else "   -  " for v in vals)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "./hest1k_datasets"
    files = sorted(glob.glob(os.path.join(root, "*", "results", "reproduce_*.csv")))
    if not files:
        raise SystemExit(f"no reproduce_*.csv under {root}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    for organ in ["her2st", "kidney", "PRAD"]:
        d = df[(df.organ == organ) & df.gene_list.isin(PAPER_LISTS[organ])]
        if d.empty:
            continue
        k = CORE_K[organ]
        core_cols = ["core_pcc_10", "core_pcc_50", f"core_pcc_{k}"]
        g10 = [f"hv10k_pcc_{kk}" for kk in T2_KS]
        g5 = [f"hv5k_pcc_{kk}" for kk in T2_KS]
        print(f"\n{'=' * 100}\n{organ}  (test slide, Core-HVG K={k}; anchor lists {PAPER_LISTS[organ]}; "
              f"averaged over lists)\n{'=' * 100}")
        for bb in ["UNI", "Exaone", "CONCH"]:
            b = d[d.backbone == bb]
            if b.empty:
                continue
            init = b[b.config == "init"][core_cols].mean().to_numpy()
            gr = b[b.config == "generag"]
            print(f"\n[{bb}]  Core HVG  {'PCC-10  PCC-50  PCC-'+str(k):>22}")
            if (organ, bb, False) in TABLE1:
                print(f"  paper  backbone only : {fmt(TABLE1[(organ, bb, False)])}")
            print(f"  ours   backbone only : {fmt(init)}")
            if gr.empty:
                continue
            if (organ, bb, True) in TABLE1:
                print(f"  paper  + GeneRAG     : {fmt(TABLE1[(organ, bb, True)])}")
            fixed = gr[(gr.alpha == DEFAULT_CFG["alpha"]) & (gr.omega == DEFAULT_CFG["omega"])]
            if not fixed.empty:
                print(f"  ours   + GeneRAG a=0.01 w=0.75 : {fmt(fixed[core_cols].mean().to_numpy())}")
            # Best config chosen on Core PCC-10 (mean over lists), then reported.
            agg = gr.groupby(["alpha", "omega"])[core_cols + g10 + g5 + ["sparsity"]].mean()
            best_core = agg["core_pcc_10"].idxmax()
            print(f"  ours   + GeneRAG best(core) a={best_core[0]} w={best_core[1]} : "
                  f"{fmt(agg.loc[best_core, core_cols])}   (nnz={agg.loc[best_core, 'sparsity']:.1f})")

            print(f"  Global HVG {'PCC-'+' PCC-'.join(map(str, T2_KS))}")
            if (organ, bb) in TABLE2:
                print(f"  paper  + GeneRAG                : {fmt(TABLE2[(organ, bb)])}")
            best_g = agg["hv10k_pcc_10"].idxmax()
            print(f"  ours   best(global) a={best_g[0]} w={best_g[1]} 10k panel : {fmt(agg.loc[best_g, g10])}")
            print(f"  ours   best(global) a={best_g[0]} w={best_g[1]} 5k panel  : {fmt(agg.loc[best_g, g5])}")
            if not fixed.empty:
                print(f"  ours   a=0.01 w=0.75          10k panel : {fmt(fixed[g10].mean().to_numpy())}")

        # Full sweep table (mean over anchor lists) for reference.
        print(f"\n  sweep (mean over lists) — Core PCC-10 by (alpha × omega):")
        gr = d[d.config == "generag"]
        for bb in gr.backbone.unique():
            piv = gr[gr.backbone == bb].pivot_table(index="alpha", columns="omega", values="core_pcc_10", aggfunc="mean")
            print(f"  [{bb}]\n{piv.round(4).to_string()}")


if __name__ == "__main__":
    main()
