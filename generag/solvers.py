# -*- coding: utf-8 -*-
"""
Sparse-regression solvers for the GeneRAG inference objective.

Two backends are supported:

* **GPU (PyTorch + CUDA)** — batched FISTA for Lasso / ElasticNet and a
  closed-form solve for Ridge. Used by default on CUDA-capable devices for
  the methods listed in :data:`GPU_METHODS`.
* **CPU (scikit-learn)** — coordinate-descent / LARS / OMP / NNLS / Bayesian
  ridge. Used as a fallback when the GPU backend is unavailable or when the
  requested method is not implemented on GPU.

The two backends share the same loss definitions and produce numerically
equivalent solutions; see ``examples/benchmark.py`` for empirical
verification (Pearson r between the two outputs is ~0.999 on real data).
"""

from __future__ import annotations

import numpy as np
import torch


# Methods that have an optimized GPU implementation in this module.
GPU_METHODS = ("lasso", "ridge", "elasticnet")


# =============================================================================
# GPU primitives
# =============================================================================

def _power_iter_op_norm_sq(D: torch.Tensor, n_iter: int = 30) -> float:
    """Estimate the squared spectral norm of ``D`` via power iteration.

    This gives the Lipschitz constant of the gradient of (1/2)||DW-B||^2,
    which sets the FISTA step size.
    """
    N = D.shape[1]
    v = torch.randn(N, device=D.device, dtype=D.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(n_iter):
        v = D.T @ (D @ v)
        nv = v.norm()
        if nv < 1e-20:
            return 0.0
        v = v / nv
    Dv = D @ v
    return float((Dv @ Dv).item())


def _soft_threshold(x: torch.Tensor, t: float) -> torch.Tensor:
    """Element-wise soft-threshold (proximal operator of t * L1 norm)."""
    return torch.sign(x) * torch.clamp(x.abs() - t, min=0.0)


def _solve_ridge_gpu(
    D: torch.Tensor,
    B: torch.Tensor,
    alpha: float,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Closed-form Ridge solve in scikit-learn's loss convention.

    Minimizes ``||DW - B||^2 + alpha * ||W||^2``  ->  W = (DᵀD + αI)⁻¹ DᵀB.
    """
    Dt = D.to(dtype)
    Bt = B.to(dtype)
    A = Dt.T @ Dt
    A.diagonal().add_(alpha)
    return torch.linalg.solve(A, Dt.T @ Bt)


def _fista_batch(
    D: torch.Tensor,
    B: torch.Tensor,
    l1_strength: float,
    l2_strength: float = 0.0,
    positive: bool = False,
    max_iter: int = 2000,
    tol: float = 1e-4,
    check_every: int = 25,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Batched FISTA for an ElasticNet-style objective.

    Solves (jointly across all columns of ``B``):

        min_W (1 / (2 * M)) * ||DW - B||_F^2
              + (l2_strength / 2) * ||W||_F^2
              + l1_strength * ||W||_{1,1}
              [+ indicator(W >= 0) if positive=True]

    where the first term matches scikit-learn's per-sample normalization
    (``M`` = number of rows of ``D``).
    """
    D = D.to(dtype)
    B = B.to(dtype)
    M, N = D.shape
    K = B.shape[1]
    n_samples = float(M)

    # Lipschitz constant of the smooth part's gradient.
    op_sq = _power_iter_op_norm_sq(D)
    L = op_sq / n_samples + l2_strength
    step = 1.0 / max(L, 1e-12)
    thresh = step * l1_strength

    W = torch.zeros(N, K, device=D.device, dtype=dtype)
    Z = W.clone()
    t = 1.0

    for it in range(max_iter):
        W_prev = W
        # Gradient of the smooth part at the momentum point Z.
        grad = (D.T @ (D @ Z - B)) / n_samples
        if l2_strength != 0.0:
            grad = grad + l2_strength * Z
        V = Z - step * grad

        # Proximal step.
        if l1_strength > 0:
            if positive:
                W = torch.clamp(V - thresh, min=0.0)
            else:
                W = _soft_threshold(V, thresh)
        else:
            W = torch.clamp(V, min=0.0) if positive else V

        # Nesterov momentum update.
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        Z = W + ((t - 1.0) / t_new) * (W - W_prev)
        t = t_new

        # Relative-change stopping criterion.
        if (it + 1) % check_every == 0:
            num = (W - W_prev).abs().max()
            den = W.abs().max().clamp_min(1e-12)
            if float(num / den) < tol:
                break

    return W


def _solve_gpu(
    D_np: np.ndarray,
    B_np: np.ndarray,
    method: str,
    params: dict,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """GPU dispatcher. Returns ``coef_`` with shape ``(n_targets, n_features)``."""
    dev = torch.device(device)
    D = torch.from_numpy(D_np).to(dev, dtype=dtype, non_blocking=True)
    B = torch.from_numpy(B_np).to(dev, dtype=dtype, non_blocking=True)

    p = params or {}
    max_iter = int(p.get("max_iter", 2000))
    tol = float(p.get("tol", 1e-4))
    positive = bool(p.get("positive", True))

    if method == "ridge":
        W = _solve_ridge_gpu(D, B, alpha=float(p.get("alpha", 1.0)), dtype=torch.float64)
    elif method == "lasso":
        W = _fista_batch(
            D, B,
            l1_strength=float(p.get("alpha", 0.01)),
            l2_strength=0.0,
            positive=positive, max_iter=max_iter, tol=tol, dtype=dtype,
        )
    elif method == "elasticnet":
        alpha = float(p.get("alpha", 0.01))
        l1_ratio = float(p.get("l1_ratio", 0.5))
        W = _fista_batch(
            D, B,
            l1_strength=alpha * l1_ratio,
            l2_strength=alpha * (1.0 - l1_ratio),
            positive=positive, max_iter=max_iter, tol=tol, dtype=dtype,
        )
    else:
        raise ValueError(f"GPU backend does not support method='{method}'")

    # (N, K) -> (K, N) to match sklearn's coef_ convention.
    return W.T.contiguous().cpu().numpy()


# =============================================================================
# CPU (scikit-learn) fallback
# =============================================================================

def _solve_sklearn(
    D_np: np.ndarray,
    B_np: np.ndarray,
    method: str,
    params: dict,
) -> np.ndarray:
    """scikit-learn dispatcher. Returns ``coef_`` with shape ``(n_targets, n_features)``.

    Supports the GPU-backed methods plus several CPU-only methods
    (``omp``, ``nnls``, ``lassolars``, ``bayesian_ridge``).
    """
    p = params or {}

    if method == "lasso":
        from sklearn.linear_model import Lasso
        model = Lasso(
            alpha=p.get("alpha", 0.01),
            fit_intercept=False,
            positive=p.get("positive", True),
            max_iter=p.get("max_iter", 2000),
        )
        model.fit(D_np, B_np)
        return np.atleast_2d(model.coef_)

    if method == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=p.get("alpha", 1.0), fit_intercept=False)
        model.fit(D_np, B_np)
        return np.atleast_2d(model.coef_)

    if method == "elasticnet":
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(
            alpha=p.get("alpha", 0.01),
            l1_ratio=p.get("l1_ratio", 0.5),
            fit_intercept=False,
            positive=p.get("positive", True),
            max_iter=p.get("max_iter", 2000),
        )
        model.fit(D_np, B_np)
        return np.atleast_2d(model.coef_)

    if method == "omp":
        from sklearn.linear_model import OrthogonalMatchingPursuit
        model = OrthogonalMatchingPursuit(
            n_nonzero_coefs=p.get("n_nonzero_coefs"),
            tol=p.get("tol"),
            fit_intercept=False,
        )
        model.fit(D_np, B_np)
        return np.atleast_2d(model.coef_)

    if method == "lassolars":
        from sklearn.linear_model import LassoLars
        model = LassoLars(
            alpha=p.get("alpha", 0.01),
            fit_intercept=False,
            positive=p.get("positive", True),
        )
        model.fit(D_np, B_np)
        return np.atleast_2d(model.coef_)

    if method == "bayesian_ridge":
        from sklearn.linear_model import BayesianRidge
        model = BayesianRidge(
            alpha_1=p.get("alpha_1", 1e-6),
            alpha_2=p.get("alpha_2", 1e-6),
            lambda_1=p.get("lambda_1", 1e-6),
            lambda_2=p.get("lambda_2", 1e-6),
            fit_intercept=False,
        )
        model.fit(D_np, B_np)
        return np.atleast_2d(model.coef_)

    if method == "nnls":
        # scipy NNLS is single-target; loop across columns of B.
        from scipy.optimize import nnls
        if B_np.ndim == 1:
            coef, _ = nnls(D_np, B_np)
            return np.atleast_2d(coef)
        coefs = np.stack([nnls(D_np, B_np[:, k])[0] for k in range(B_np.shape[1])], axis=0)
        return coefs

    raise ValueError(
        f"Unknown method='{method}'. Use one of: "
        "lasso, ridge, elasticnet, omp, lassolars, bayesian_ridge, nnls."
    )


# =============================================================================
# Public entry point
# =============================================================================

def solve_batch(
    D: np.ndarray,
    B: np.ndarray,
    method: str = "elasticnet",
    params: dict | None = None,
    device: str = "cuda",
) -> np.ndarray:
    """Solve a (multi-output) sparse regression and return ``coef_``.

    Dispatches to the GPU backend when ``device`` requests CUDA and the
    method is GPU-supported; otherwise falls back to scikit-learn.

    Parameters
    ----------
    D : ndarray of shape (M, N)
        Design matrix (already pre-scaled if used inside GeneRAG).
    B : ndarray of shape (M, K)
        Right-hand side. ``K`` is the number of test spots.
    method : str
        One of ``{'lasso', 'ridge', 'elasticnet', 'omp', 'lassolars',
        'bayesian_ridge', 'nnls'}``. The first three have a GPU
        implementation.
    params : dict, optional
        Method-specific hyperparameters (e.g. ``alpha``, ``l1_ratio``,
        ``positive``, ``max_iter``, ``tol``).
    device : str
        ``'cuda'`` / ``'cuda:0'`` to request the GPU backend, anything else
        (including ``'cpu'``) to force the scikit-learn fallback.

    Returns
    -------
    coef : ndarray of shape (K, N)
        Solution matrix, one row per test spot. Matches scikit-learn's
        multi-output ``coef_`` convention.
    """
    params = params or {}
    D = np.ascontiguousarray(D)
    B = np.ascontiguousarray(B)

    use_gpu = (
        method in GPU_METHODS
        and isinstance(device, str) and device.startswith("cuda")
        and torch.cuda.is_available()
    )
    if use_gpu:
        return _solve_gpu(D, B, method, params, device).astype(np.float64)
    return _solve_sklearn(D, B, method, params).astype(np.float64)
