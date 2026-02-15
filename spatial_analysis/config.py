# -*- coding: utf-8 -*-
"""
Configuration Module

실험 설정 및 상수들을 정의합니다.
"""

# 최적화 실험을 위한 search space
search_space = {
    # 1. Lasso: 10이 최적이라는 단서를 기준으로 좁고 조밀하게, 그리고 넓게 탐색
    # embedding_ratio: [0,1]. 0=유전자만, 1=임베딩만, 0.5=동일 비율 (gene_weight + embedding_weight = 1)
    'lasso': {
        'alpha': [0.001, 0.01, 0.1, 1, 5],
        'fit_intercept': [True],
        'embedding_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
    },

    # 2. ElasticNet: alpha는 Lasso와 비슷하게 가져가되, l1_ratio로 비율 조절
    # l1_ratio가 1에 가까우면 Lasso(10)와 비슷해지므로 alpha 범위도 10 주변 포함
    'elasticnet': {
        'alpha': [0.001, 0.01, 0.1, 1, 5],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9],  # 1.0은 Lasso와 동일
        'embedding_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
    },

    # 3. Ridge: L2 usually needs larger alpha (e.g. 10-1000)
    'ridge': {
        'alpha': [1, 10, 50],
        'solver': ['auto']  # solver constraint when positive=True
    },
}
