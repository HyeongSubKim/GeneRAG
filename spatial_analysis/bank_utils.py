# -*- coding: utf-8 -*-
"""
Bank Utilities Module

Bank 데이터 관련 유틸리티 함수들을 제공합니다.
"""

import numpy as np
import pandas as pd


def select_high_variable_genes(bank_all_df, n_genes=3000):
    """
    Bank 데이터에서 분산이 높은 상위 n_genes개 유전자 선택
    
    Parameters:
    -----------
    bank_all_df: DataFrame (spot x all_genes) - Bank 전체 데이터
    n_genes: int - 선택할 유전자 수 (기본값: 3000)
    
    Returns:
    --------
    high_var_genes: list - 선택된 high variable genes 이름 리스트
    """
    # 각 유전자의 분산 계산
    gene_variance = bank_all_df.var(axis=0)  # 각 열(유전자)의 분산
    
    # 상위 n_genes개 유전자 선택
    top_genes = gene_variance.nlargest(n_genes)
    high_var_genes = top_genes.index.tolist()
    
    print(f"  - High variable genes selected: {len(high_var_genes)}")
    print(f"    variance range: [{top_genes.min():.4f}, {top_genes.max():.4f}]")
    
    return high_var_genes


def prepare_bank_data(bank_all_df, high_var_genes=None):
    """
    bank_all_df를 (genes x spots) 형태의 numpy array로 변환
    high_var_genes가 제공되면 해당 유전자만 필터링
    
    Parameters:
    -----------
    bank_all_df: DataFrame (spot x all_genes) - Bank 전체 데이터
    high_var_genes: list - 선택할 유전자 이름 리스트 (None이면 전체 유전자 사용)
    
    Returns:
    --------
    bank_full: numpy array (M_total, N_spots) - 행=유전자, 열=spot
    gene_names: list - 유전자 이름 리스트
    """
    # High variable genes 필터링
    if high_var_genes is not None:
        # 공통 유전자만 선택 (bank에 없는 유전자 제외)
        common_genes = [g for g in high_var_genes if g in bank_all_df.columns]
        if len(common_genes) < len(high_var_genes):
            print(f"  Warning: {len(high_var_genes) - len(common_genes)} genes not in bank.")
        bank_filtered = bank_all_df[common_genes]
        gene_names = common_genes
    else:
        bank_filtered = bank_all_df
        gene_names = bank_all_df.columns.tolist()
    
    # Transpose: (spot x genes) -> (genes x spots)
    bank_full = bank_filtered.T.values  # Shape: (M_total, N_spots)
    
    return bank_full, gene_names


def get_gene_indices(test_pred_genes, bank_gene_names):
    """
    test_pred_df의 column명을 사용하여 bank_all_df에서 유전자 인덱스 매핑 생성
    
    Parameters:
    -----------
    test_pred_genes: list - test_pred_df의 column명 (300개 유전자 이름)
    bank_gene_names: list - bank_all_df의 column명 (전체 유전자 이름)
    
    Returns:
    --------
    gene_indices: numpy array (M_obs,) - 300개 유전자의 전체 유전자 인덱스
    valid_genes: list - bank에 존재하는 유전자 이름 리스트
    """
    # 공통 유전자 찾기
    valid_genes = [gene for gene in test_pred_genes if gene in bank_gene_names]
    
    if len(valid_genes) == 0:
        raise ValueError("No common genes found.")
    
    if len(valid_genes) < len(test_pred_genes):
        print(f"Warning: {len(test_pred_genes) - len(valid_genes)} genes not in bank.")
    
    # 유전자 인덱스 매핑
    gene_to_idx = {gene: idx for idx, gene in enumerate(bank_gene_names)}
    gene_indices = np.array([gene_to_idx[gene] for gene in valid_genes])
    
    return gene_indices, valid_genes


def find_best_bank_spots(test_spot_pred, bank_selected_df, k=5):
    """
    Test spot과 가장 높은 correlation을 가진 k개의 bank spot 찾기
    
    Parameters:
    -----------
    test_spot_pred: Series 또는 DataFrame - Test spot의 유전자 발현량
    bank_selected_df: DataFrame (spot x selected_genes) - Bank selected genes 데이터
    k: int - 찾을 bank spot 개수 (기본값: 5)
    
    Returns:
    --------
    best_indices: 상위 k개 bank spot 인덱스
    correlations: correlation 값들
    """
    # bank_selected_df가 가지고 있는 유전자만 사용
    bank_genes = bank_selected_df.columns.tolist()
    
    # test_spot_pred에서 해당 유전자만 추출 (공통 유전자만 사용)
    if isinstance(test_spot_pred, pd.Series):
        # 공통 유전자만 필터링
        common_genes = [g for g in bank_genes if g in test_spot_pred.index]
        if len(common_genes) == 0:
            # 공통 유전자가 없으면 모든 bank spot에 대해 NaN 반환
            correlations = np.full(len(bank_selected_df), np.nan)
            best_indices = np.argsort(correlations)[::-1][:k]
            return best_indices, correlations[best_indices]
        test_spot_filtered = test_spot_pred[common_genes].values
        bank_genes_to_use = common_genes
    else:
        # DataFrame인 경우
        common_genes = [g for g in bank_genes if g in test_spot_pred.columns]
        if len(common_genes) == 0:
            correlations = np.full(len(bank_selected_df), np.nan)
            best_indices = np.argsort(correlations)[::-1][:k]
            return best_indices, correlations[best_indices]
        test_spot_filtered = test_spot_pred[common_genes].values.flatten()
        bank_genes_to_use = common_genes
    
    # Test spot과 각 bank spot 간 correlation 계산
    correlations = []
    
    for idx in range(len(bank_selected_df)):
        bank_spot = bank_selected_df.iloc[idx][bank_genes_to_use].values
        
        # NaN 처리
        mask = ~(np.isnan(bank_spot) | np.isnan(test_spot_filtered))
        if mask.sum() < 3:  # 최소 3개 유전자 필요
            corr = np.nan
        else:
            corr = np.corrcoef(bank_spot[mask], test_spot_filtered[mask])[0, 1]
        
        correlations.append(corr if not np.isnan(corr) else -1.0)
    
    correlations = np.array(correlations)
    
    # 상위 k개 선택
    best_indices = np.argsort(correlations)[::-1][:k]
    
    return best_indices, correlations[best_indices]
