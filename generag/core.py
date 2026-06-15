# -*- coding: utf-8 -*-
"""
GeneRAG — plug-and-play retrieval-augmented gene expression predictor.

Typical usage
-------------

>>> from generag import GeneRAG
>>> model = GeneRAG(
...     bank_expression=bank_df,           # (n_bank_spots, n_genes)
...     bank_embeddings=bank_embeddings,   # (n_bank_spots, d) — optional
...     anchor_genes=anchor_gene_list,
...     n_high_variable_genes=10_000,
... )
>>> predictions, mean_sparsity = model.predict(
...     test_anchor=test_pred_df,          # (n_test_spots, n_anchor_genes)
...     test_embeddings=test_embeddings,   # (n_test_spots, d) — optional
...     method='elasticnet',
...     alpha=0.01, l1_ratio=0.9,
...     embedding_ratio=0.75, positive=True,
...     device='cuda',
... )

The model is built once from a reference bank and can be queried many times
with new test inputs — this is the "plug-and-play" surface of GeneRAG.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .data import (
    select_high_variable_genes,
    prepare_bank_dictionary,
    get_gene_index_map,
)
from .solvers import solve_batch


class GeneRAG:
    """Retrieval-augmented gene expression predictor.

    The model stores a reference bank dictionary ``D ∈ R^{G_hv × N}``
    (genes × bank spots) together with (optionally) per-spot visual
    embeddings, and reconstructs full gene-expression profiles for new
    spots by solving a multi-output ElasticNet over ``D``.

    Parameters
    ----------
    bank_expression : pandas.DataFrame
        Spot × gene expression matrix used as the retrieval bank. Index
        entries identify spots; columns are gene names.
    bank_embeddings : ndarray of shape (n_bank_spots, d), optional
        Per-spot visual embeddings aligned with ``bank_expression.index``.
        Required only if ``embedding_ratio > 0`` will be used at predict
        time.
    anchor_genes : sequence of str, optional
        Anchor gene subset used as one component of the regression target.
        Genes not present in the bank are silently dropped. If ``None``,
        all bank genes are treated as candidate anchors and the actual
        anchor set is determined at predict time by the columns of
        ``test_anchor``.
    n_high_variable_genes : int, default 10000
        Number of top-variance bank genes retained in the dictionary
        ``D``. Limits both memory and reconstruction dimensionality.
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        bank_expression: pd.DataFrame,
        bank_embeddings: np.ndarray | None = None,
        anchor_genes: Sequence[str] | None = None,
        n_high_variable_genes: int = 10_000,
    ):
        if bank_embeddings is not None:
            bank_embeddings = np.ascontiguousarray(bank_embeddings)
            if bank_embeddings.shape[0] != len(bank_expression):
                raise ValueError(
                    f"bank_embeddings has {bank_embeddings.shape[0]} rows but "
                    f"bank_expression has {len(bank_expression)} rows."
                )

        # Pick the top-variance genes that will form the rows of D.
        hv_genes = select_high_variable_genes(bank_expression, n_genes=n_high_variable_genes)
        D_full, gene_names = prepare_bank_dictionary(bank_expression, high_var_genes=hv_genes)

        # Map each anchor gene to a row index of D.
        anchor_genes = list(anchor_genes) if anchor_genes is not None else gene_names
        anchor_indices, valid_anchors = get_gene_index_map(anchor_genes, gene_names)

        self._D_full: np.ndarray = D_full                       # (G_hv, N)
        self._gene_names: list[str] = gene_names                # length G_hv
        self._anchor_indices: np.ndarray = anchor_indices       # length |valid_anchors|
        self._valid_anchors: list[str] = valid_anchors          # subset of anchor_genes that hit D
        self._bank_embeddings: np.ndarray | None = bank_embeddings  # (N, d) or None
        self._bank_spot_index: pd.Index = bank_expression.index

    # --------------------------------------------------------------- helpers

    @property
    def gene_names(self) -> list[str]:
        """Gene names corresponding to the rows of the bank dictionary D."""
        return list(self._gene_names)

    @property
    def anchor_genes(self) -> list[str]:
        """Anchor genes that were successfully mapped into D."""
        return list(self._valid_anchors)

    @property
    def n_bank_spots(self) -> int:
        return self._D_full.shape[1]

    @property
    def n_genes(self) -> int:
        return self._D_full.shape[0]

    def __repr__(self) -> str:  # pragma: no cover
        emb = "yes" if self._bank_embeddings is not None else "no"
        return (
            f"GeneRAG(bank_spots={self.n_bank_spots}, n_genes={self.n_genes}, "
            f"anchors={len(self._valid_anchors)}, embeddings={emb})"
        )

    # ---------------------------------------------------------------- predict

    def predict(
        self,
        test_anchor: pd.DataFrame,
        test_embeddings: np.ndarray | None = None,
        *,
        method: str = "elasticnet",
        alpha: float = 0.01,
        l1_ratio: float = 0.9,
        embedding_ratio: float = 0.0,
        positive: bool = True,
        max_iter: int = 2000,
        tol: float = 1e-4,
        device: str = "cuda",
        return_sparsity: bool = True,
    ) -> tuple[pd.DataFrame, float] | pd.DataFrame:
        """Reconstruct full gene-expression profiles for new test spots.

        Builds the dual-modality design matrix from anchor genes and
        embeddings, solves a single multi-output sparse regression jointly
        over all test spots, and reconstructs ``D @ α`` for each spot.

        Parameters
        ----------
        test_anchor : pandas.DataFrame
            (n_test_spots, n_anchor_genes) target matrix on the anchor
            modality. Columns must contain the anchor genes. NaNs are not
            supported in the joint solve path.
        test_embeddings : ndarray of shape (n_test_spots, d), optional
            Visual embeddings of the test spots. Required when
            ``embedding_ratio > 0``.
        method : str, default 'elasticnet'
            Sparse-regression method passed to :func:`generag.solvers.solve_batch`.
        alpha, l1_ratio, positive, max_iter, tol
            Standard ElasticNet / Lasso / Ridge hyperparameters.
        embedding_ratio : float in [0, 1], default 0.0
            Weighting between the two modalities. ``0`` uses anchors only,
            ``1`` uses embeddings only. Concretely the model solves
            ``(1-ω)·||y_anchor - D_anchor α||² + ω·||f_img - D_img α||² + reg(α)``.
        device : str, default 'cuda'
            ``'cuda'`` / ``'cuda:0'`` to use the GPU backend (recommended);
            ``'cpu'`` forces the scikit-learn fallback.
        return_sparsity : bool, default True
            If True, also returns the mean number of non-zero coefficients
            per test spot.

        Returns
        -------
        predicted : pandas.DataFrame
            (n_test_spots, n_genes) reconstructed expression matrix.
            Columns are :attr:`gene_names`; index inherits from
            ``test_anchor.index``.
        mean_sparsity : float
            Mean ``count_nonzero(α)`` across test spots (only when
            ``return_sparsity=True``).
        """
        embedding_ratio = float(np.clip(embedding_ratio, 0.0, 1.0))
        gene_weight = 1.0 - embedding_ratio
        embedding_weight = embedding_ratio

        use_gene = gene_weight > 0
        use_emb = embedding_weight > 0 and test_embeddings is not None and self._bank_embeddings is not None

        if not use_gene and not use_emb:
            raise ValueError(
                "Both modalities are disabled. Provide bank/test embeddings or set "
                "embedding_ratio < 1."
            )
        if embedding_ratio >= 1.0 and not use_emb:
            raise ValueError(
                "embedding_ratio=1 requires both bank_embeddings (at construction time) "
                "and test_embeddings."
            )

        # ----- Build augmented design matrix D_sub and target B -----
        D_sub_parts: list[np.ndarray] = []
        B_parts: list[np.ndarray] = []

        if use_gene:
            # Anchor-side block: rows of D restricted to valid anchor genes,
            # rescaled to match the (1 - ω) weight in the joint objective.
            anchor_cols = self._valid_anchors
            if not all(g in test_anchor.columns for g in anchor_cols):
                missing = [g for g in anchor_cols if g not in test_anchor.columns]
                raise ValueError(
                    f"test_anchor is missing {len(missing)} anchor genes; "
                    f"first few: {missing[:5]}"
                )
            Y = test_anchor[anchor_cols].to_numpy(dtype=np.float64)
            if np.isnan(Y).any():
                raise ValueError(
                    "test_anchor contains NaNs; the batched solver requires a complete target."
                )
            scale_g = np.sqrt(gene_weight)
            D_sub_parts.append(scale_g * self._D_full[self._anchor_indices, :])
            B_parts.append(scale_g * Y.T)  # (n_anchors, n_test)

        if use_emb:
            # Embedding-side block: embeddings act as additional rows of D,
            # rescaled to ω.
            E = np.asarray(self._bank_embeddings)             # (N, d)
            te = np.asarray(test_embeddings, dtype=np.float64)  # (n_test, d)
            if te.shape[1] != E.shape[1]:
                raise ValueError(
                    f"test_embeddings have dim {te.shape[1]} but bank_embeddings have dim {E.shape[1]}."
                )
            scale_e = np.sqrt(embedding_weight)
            D_sub_parts.append(scale_e * E.T)   # (d, N)
            B_parts.append(scale_e * te.T)      # (d, n_test)

        D_sub = np.vstack(D_sub_parts) if len(D_sub_parts) > 1 else D_sub_parts[0]
        B = np.vstack(B_parts) if len(B_parts) > 1 else B_parts[0]

        # ----- Solve the joint multi-output sparse regression -----
        params = {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "positive": positive,
            "max_iter": max_iter,
            "tol": tol,
        }
        coefs = solve_batch(D_sub, B, method=method, params=params, device=device)
        if coefs.ndim == 1:
            coefs = coefs.reshape(1, -1)

        # ----- Reconstruct full expression on the HV gene panel -----
        # coefs: (n_test, N) ; D_full: (G_hv, N) -> reconstruction (n_test, G_hv)
        reconstructed = coefs @ self._D_full.T
        predicted = pd.DataFrame(reconstructed, index=test_anchor.index, columns=self._gene_names)

        if return_sparsity:
            mean_sparsity = float(np.count_nonzero(coefs, axis=1).mean())
            return predicted, mean_sparsity
        return predicted
