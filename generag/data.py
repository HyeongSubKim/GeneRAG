# -*- coding: utf-8 -*-
"""
Data loading and bank preparation utilities.

This module groups two related sets of helpers:

* **Bank construction**: choosing high-variance genes, materializing the
  bank dictionary ``D`` (genes × spots), and mapping anchor gene names to
  row indices of ``D``.
* **Convenience I/O**: loading ``.h5ad`` slides, loading per-slide visual
  embeddings, loading test-spot predictions stored as ``.pt`` tensors, and
  loading ground-truth expression.

The I/O helpers are convenience-only; the :class:`generag.GeneRAG`
class itself accepts plain in-memory ``pandas.DataFrame`` / ``ndarray``
inputs and is fully agnostic to how those were produced.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd


# =============================================================================
# Bank construction
# =============================================================================

def select_high_variable_genes(bank_expression: pd.DataFrame, n_genes: int = 10_000) -> list[str]:
    """Return the top-``n_genes`` highest-variance gene names.

    Variance is computed column-wise on the input expression matrix.
    """
    variance = bank_expression.var(axis=0)
    top = variance.nlargest(min(n_genes, len(variance)))
    return top.index.tolist()


def prepare_bank_dictionary(
    bank_expression: pd.DataFrame,
    high_var_genes: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Materialize the bank dictionary ``D`` of shape ``(n_genes, n_spots)``.

    Genes that are requested but absent from ``bank_expression`` are dropped
    with a printed warning.
    """
    if high_var_genes is not None:
        present = [g for g in high_var_genes if g in bank_expression.columns]
        if len(present) < len(high_var_genes):
            dropped = len(high_var_genes) - len(present)
            print(f"[prepare_bank_dictionary] dropped {dropped} requested genes not in bank.")
        filtered = bank_expression[present]
        gene_names = present
    else:
        filtered = bank_expression
        gene_names = bank_expression.columns.tolist()

    # (spots, genes) -> (genes, spots) for matmul friendliness.
    D = filtered.T.to_numpy()
    return D, gene_names


def get_gene_index_map(
    query_genes: Sequence[str],
    bank_gene_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Map ``query_genes`` to row indices into the bank dictionary.

    Returns
    -------
    indices : ndarray of int
        Positions of the queryable genes inside ``bank_gene_names``.
    valid_genes : list of str
        Subset of ``query_genes`` actually present in the bank.
    """
    bank_set = set(bank_gene_names)
    valid_genes = [g for g in query_genes if g in bank_set]
    if not valid_genes:
        raise ValueError("No requested genes are present in the bank.")
    if len(valid_genes) < len(query_genes):
        print(f"[get_gene_index_map] {len(query_genes) - len(valid_genes)} requested genes missing from bank.")
    name_to_idx = {g: i for i, g in enumerate(bank_gene_names)}
    indices = np.array([name_to_idx[g] for g in valid_genes], dtype=int)
    return indices, valid_genes


# =============================================================================
# I/O helpers (anndata + torch)
# =============================================================================

def _to_dense(X) -> np.ndarray:
    """Convert a scipy sparse matrix or ndarray to a dense ndarray."""
    from scipy.sparse import issparse
    return X.toarray() if issparse(X) else np.asarray(X)


def load_gene_list(filepath: str) -> list[str]:
    """Read a newline-separated gene list from a text file."""
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_bank_from_h5ad(
    train_slides: Sequence[str],
    selected_genes: Sequence[str],
    st_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate per-slide ``.h5ad`` expression matrices.

    Returns
    -------
    bank_selected_df : DataFrame
        (n_bank_spots, len(selected_genes)) restricted to anchor genes.
    bank_all_df : DataFrame
        (n_bank_spots, n_all_genes) full gene panel, used to derive the HV
        gene dictionary inside :class:`generag.GeneRAG`.
    """
    import anndata
    from tqdm import tqdm

    selected_list, all_list = [], []
    for slide in tqdm(train_slides, desc="Loading bank slides"):
        fpath = os.path.abspath(os.path.join(st_path, slide + ".h5ad"))
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Slide file not found: {fpath}")
        adata = anndata.read_h5ad(fpath)
        index = [f"{slide}_{i}" for i in range(adata.shape[0])]

        selected_list.append(pd.DataFrame(
            _to_dense(adata[:, list(selected_genes)].X),
            columns=list(selected_genes),
            index=index,
        ))
        all_list.append(pd.DataFrame(
            _to_dense(adata.X),
            columns=adata.var_names,
            index=index,
        ))

    bank_selected_df = pd.concat(selected_list, axis=0)
    bank_all_df = pd.concat(all_list, axis=0)
    return bank_selected_df, bank_all_df


def load_test_predictions(
    pred_path: str,
    test_slide: str,
    selected_genes: Sequence[str],
    st_path: str,
    num_rep: int = 20,
) -> tuple[pd.DataFrame, list[str]]:
    """Load test-spot anchor-gene predictions stored as a ``.pt`` tensor.

    The expected file layout is ``(n_spots * num_rep, n_anchor_genes)``;
    samples for the same spot are averaged together (see GeneRAG paper §X
    for the multi-sample averaging rationale).
    """
    import anndata
    import torch

    pred = torch.load(pred_path)
    if pred.ndim == 3 and pred.shape[1] == 1:
        pred = pred.squeeze(1)

    fpath = os.path.abspath(os.path.join(st_path, test_slide + ".h5ad"))
    adata = anndata.read_h5ad(fpath)
    n_spots = adata.shape[0]

    pred_avg = torch.zeros(n_spots, pred.shape[1])
    for i in range(n_spots):
        sample_idx = np.random.choice(num_rep, num_rep, replace=False)
        pred_avg[i] = torch.mean(pred[i * num_rep + sample_idx, :], dim=0)

    cols = list(selected_genes[:pred_avg.shape[1]])
    df = pd.DataFrame(pred_avg.cpu().numpy(), columns=cols, index=adata.obs_names[:n_spots])
    return df, df.index.tolist()


def load_ground_truth(
    test_spots: Sequence[str],
    test_slide: str,
    st_path: str,
    log2: bool = True,
) -> pd.DataFrame:
    """Load the log2(x+1)-transformed ground-truth expression for ``test_slide``."""
    import anndata
    fpath = os.path.abspath(os.path.join(st_path, test_slide + ".h5ad"))
    adata = anndata.read_h5ad(fpath)
    df = pd.DataFrame(
        _to_dense(adata.X),
        columns=adata.var_names,
        index=list(test_spots),
    )
    return np.log2(df + 1) if log2 else df


def load_slide_embedding(
    embed_dir: str,
    slide_id: str,
    file_suffix: str = "_uni_aug.pt",
) -> np.ndarray | None:
    """Load a single slide's ``.pt`` embedding; returns ``None`` if missing.

    A 3D tensor of shape ``(n_spots, n_aug, dim)`` is collapsed to
    ``(n_spots, dim)`` by averaging over the augmentation axis.
    """
    import torch
    fpath = os.path.join(embed_dir, slide_id + file_suffix)
    if not os.path.isfile(fpath):
        return None
    emb = torch.load(fpath, map_location="cpu")
    arr = emb.cpu().numpy() if hasattr(emb, "cpu") else np.asarray(emb)
    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    return arr


def load_bank_embeddings(
    embed_dir: str,
    bank_spot_names: Sequence[str],
    file_suffix: str = "_uni_aug.pt",
) -> np.ndarray | None:
    """Assemble a bank embedding matrix aligned with ``bank_spot_names``.

    Each spot name is expected to be ``"<slide_id>_<spot_idx>"``. Returns
    ``None`` if any slide is missing from ``embed_dir``.
    """
    meta = []
    for name in bank_spot_names:
        head, _, tail = name.rpartition("_")
        if not tail.isdigit():
            return None
        meta.append((head, int(tail)))

    slides = list(dict.fromkeys(s for s, _ in meta))
    loaded: dict[str, np.ndarray] = {}
    for sid in slides:
        arr = load_slide_embedding(embed_dir, sid, file_suffix)
        if arr is None:
            return None
        loaded[sid] = arr

    dim = next(iter(loaded.values())).shape[1]
    out = np.zeros((len(meta), dim), dtype=np.float64)
    for i, (sid, idx) in enumerate(meta):
        out[i] = loaded[sid][idx]
    return out


def load_test_embeddings(
    embed_dir: str,
    test_slide: str,
    test_spots: Sequence[str],
    file_suffix: str = "_uni_aug.pt",
) -> np.ndarray | None:
    """Load the embedding for one slide and reorder rows by ``test_spots``."""
    arr = load_slide_embedding(embed_dir, test_slide, file_suffix)
    if arr is None:
        return None
    indices = []
    for name in test_spots:
        head, _, tail = name.rpartition("_")
        if tail.isdigit():
            indices.append(int(tail))
        else:
            try:
                indices.append(int(name))
            except ValueError:
                indices.append(len(indices))
    indices = np.asarray(indices)
    if indices.max() >= arr.shape[0]:
        return None
    return arr[indices]
