# -*- coding: utf-8 -*-
"""
GeneRAG — retrieval-augmented gene expression prediction from histology.

Public API
----------

>>> from generag import GeneRAG
>>> model = GeneRAG(bank_expression=bank_df, anchor_genes=anchor_list)
>>> predictions, sparsity = model.predict(test_anchor=test_df, method='elasticnet')

For paper-style hyperparameter sweeps:

>>> from generag.experiment import run_sweep
>>> results = run_sweep(
...     bank_expression=bank_df,
...     test_anchor=test_df,
...     test_gt=gt_df,
...     search_space={'elasticnet': {'alpha': [0.01], 'l1_ratio': [0.9],
...                                  'embedding_ratio': [0.0, 0.5, 0.75]}},
...     n_jobs=4,
... )

For ad-hoc evaluation of any prediction matrix:

>>> from generag.metrics import evaluate_predictions
>>> metrics = evaluate_predictions(predictions, gt_df, calibration='log1p')

For convenience I/O (loading ``.h5ad`` slides and ``.pt`` embeddings):

>>> from generag.data import (load_bank_from_h5ad, load_test_predictions,
...                           load_ground_truth, load_bank_embeddings,
...                           load_test_embeddings)
"""

__version__ = "0.1.0"

from .core import GeneRAG
from .solvers import solve_batch, GPU_METHODS
from .metrics import evaluate_predictions, gene_pearson, spot_pearson
from . import data, experiment, utils

__all__ = [
    "__version__",
    "GeneRAG",
    "solve_batch",
    "GPU_METHODS",
    "evaluate_predictions",
    "gene_pearson",
    "spot_pearson",
    "data",
    "experiment",
    "utils",
]
