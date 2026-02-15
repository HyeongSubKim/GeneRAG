# -*- coding: utf-8 -*-
"""
Spatial Transcriptomics Analysis Module

이 모듈은 H&E 이미지 기반 spatial transcriptomics 분석을 위한 함수들을 제공합니다.
"""

from .data_loading import (
    load_selected_genes,
    load_test_prediction,
    load_bank_data,
    gt_load,
    infer_test_spots,
    _to_array
)

from .generag import (
    generag_single_spot_sparse_coding_v2,
    sparse_coding_generag_v2
)

from .evaluation import (
    calculate_spot_correlation,
    calculate_gene_correlation,
    gene_correlation_analysis,
    evaluate_generag_result,
    run_optimization_experiment
)

from .bank_utils import (
    find_best_bank_spots,
    select_high_variable_genes,
    prepare_bank_data,
    get_gene_indices
)

from .visualization import (
    setup_korean_font
)

from .config import (
    search_space
)

__all__ = [
    # Data loading
    'load_selected_genes',
    'load_test_prediction',
    'load_bank_data',
    'gt_load',
    'infer_test_spots',
    '_to_array',
    # GeneRAG
    'generag_single_spot_sparse_coding_v2',
    'sparse_coding_generag_v2',
    # Evaluation
    'calculate_spot_correlation',
    'calculate_gene_correlation',
    'gene_correlation_analysis',
    'evaluate_generag_result',
    'run_optimization_experiment',
    # Bank utils
    'find_best_bank_spots',
    'select_high_variable_genes',
    'prepare_bank_data',
    'get_gene_indices',
    # Visualization
    'setup_korean_font',
    # Config
    'search_space',
]
