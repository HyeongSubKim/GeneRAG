# -*- coding: utf-8 -*-
"""
Data Loading Module

데이터 로딩 관련 함수들을 제공합니다.
"""

import os
import numpy as np
import pandas as pd
import anndata
import torch
from scipy.sparse import issparse


def _to_array(X):
    """sparse matrix 또는 numpy array를 numpy array로 변환"""
    if issparse(X):
        return X.toarray()
    else:
        return np.asarray(X)


def load_selected_genes(filepath):
    """선택된 유전자 리스트 로드"""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_test_prediction(test_slide, selected_genes, pred_path=None, st_path=None):
    """
    Test slide의 예측 결과 로드
    
    Parameters:
    -----------
    test_slide: test slide 이름
    selected_genes: 선택된 유전자 리스트
    pred_path: 예측 결과 파일 경로 (CSV 또는 .pt 파일, None이면 adata에서 로드)
    st_path: ST 데이터 경로
    
    Returns:
    --------
    test_pred_df: DataFrame (spot x selected_genes)
    test_spots: spot 이름 리스트
    """
    if pred_path is not None:
        # 파일 확장자에 따라 다른 방식으로 로드
        if pred_path.endswith('.csv'):
            # CSV 파일로 로드 (이미 log transform되어 있을 수 있음)
            test_pred_df = pd.read_csv(pred_path, index_col=0)
            # 선택된 유전자만 필터링
            common_genes = [g for g in selected_genes if g in test_pred_df.columns]
            if len(common_genes) != len(selected_genes):
                print(f"Warning: {len(selected_genes) - len(common_genes)} selected genes not in CSV.")
            test_pred_df = test_pred_df[common_genes]
        elif pred_path.endswith('.pt'):
            # PyTorch tensor 파일로 로드
            pred = torch.load(pred_path)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            # 예측 데이터를 DataFrame으로 변환
            if pred.ndim == 2:
                test_pred_df = pd.DataFrame(
                    pred,
                    columns=selected_genes[:pred.shape[1]] if len(selected_genes) >= pred.shape[1] else selected_genes,
                    index=[test_slide + "_" + str(i) for i in range(pred.shape[0])]
                )
            else:
                raise ValueError(f"Unexpected tensor shape: {pred.shape}")
        else:
            raise ValueError(f"Unsupported file format: {pred_path}")
    else:
        # Adata에서 로드 (예시 - 실제 예측값으로 교체 필요)
        if st_path is None:
            st_path = "./hest1k_datasets/her2st/st/"
        fpath = os.path.abspath(os.path.normpath(os.path.join(st_path, test_slide + ".h5ad")))
        adata = anndata.read_h5ad(fpath)
        # 실제 예측값으로 교체 필요
        X_selected = _to_array(adata[:, selected_genes].X)
        test_pred_df = pd.DataFrame(
            X_selected,
            columns=selected_genes,
            index=[test_slide + "_" + str(i) for i in range(adata.shape[0])]
        )
    
    return test_pred_df, test_pred_df.index.tolist()


def load_bank_data(train_slides, selected_genes, st_path):
    """
    Train slides (Bank) 데이터 로드
    
    Returns:
    --------
    bank_selected_df: DataFrame (spot x selected_genes)
    bank_all_df: DataFrame (spot x all_genes)
    """
    from tqdm import tqdm
    
    bank_selected_list = []
    bank_all_list = []
    
    print("Loading bank data...")
    for slide in tqdm(train_slides):
        fpath = os.path.abspath(os.path.normpath(os.path.join(st_path, slide + ".h5ad")))
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Slide file not found: {fpath}\n"
                f"Ensure st_path exists and contains .h5ad for each train slide (e.g. SPA119.h5ad, SPA120.h5ad, ...).\n"
                f"Current st_path: {os.path.abspath(os.path.normpath(st_path))!r}"
            )
        adata = anndata.read_h5ad(fpath)

        # Selected genes
        X_selected = _to_array(adata[:, selected_genes].X)
        selected_df = pd.DataFrame(
            X_selected,
            columns=selected_genes,
            index=[slide + "_" + str(i) for i in range(adata.shape[0])]
        )
        
        # All genes
        X_all = _to_array(adata.X)
        all_df = pd.DataFrame(
            X_all,
            columns=adata.var_names,
            index=[slide + "_" + str(i) for i in range(adata.shape[0])]
        )
        
        bank_selected_list.append(selected_df)
        bank_all_list.append(all_df)
    
    bank_selected_df = pd.concat(bank_selected_list, axis=0)
    bank_all_df = pd.concat(bank_all_list, axis=0)
    
    print(f"Bank data loaded: {bank_selected_df.shape[0]} spots, {bank_selected_df.shape[1]} selected genes")
    print(f"Bank all genes: {bank_all_df.shape[1]} genes")
    
    return bank_selected_df, bank_all_df


def infer_test_spots(pred_path, test_slide, selected_genes, st_path):
    """
    .pt 파일에서 test spot 예측값 로드 및 처리
    
    Parameters:
    -----------
    pred_path: .pt 파일 경로
    test_slide: test slide 이름
    selected_genes: 선택된 유전자 리스트
    st_path: ST 데이터 경로
    
    Returns:
    --------
    test_pred_df: DataFrame (spot x selected_genes)
    test_spots: spot 이름 리스트
    """
    pred = torch.load(pred_path)
    pred = pred.squeeze(1)  # (N*20, 300) -> (N*20, 300)

    # GT 데이터 로드 (spot 이름과 순서를 맞추기 위해)
    fpath = os.path.abspath(os.path.normpath(os.path.join(st_path, test_slide + ".h5ad")))
    test_adata_temp = anndata.read_h5ad(fpath)
    num_spots = test_adata_temp.shape[0]
    num_rep = 20  # 각 spot당 생성된 샘플 수
    num_selected = 20  # 평균을 내기 위해 사용할 샘플 수

    # 각 spot에 대해 샘플들의 평균 계산
    pred_avg = torch.zeros(size=(num_spots, pred.shape[1]))
    for i in range(num_spots):
        sample_indices = np.random.choice(np.arange(num_rep), num_selected, replace=False)
        pred_avg[i] = torch.mean(pred[i*num_rep + sample_indices, :], dim=0)

    pred_avg = pred_avg.cpu().detach().numpy()

    # DataFrame 생성 (인덱스는 CSV 파일과 동일하게)
    test_pred_df = pd.DataFrame(
        pred_avg,
        columns=selected_genes[:pred_avg.shape[1]],
        index=test_adata_temp.obs_names[:num_spots]
    )

    test_spots = test_pred_df.index.tolist()

    print(f"Test prediction loaded.")
    print(f"Test spots: {len(test_spots)}")
    print(f"Test prediction shape: {test_pred_df.shape}")
    print(f"Test prediction index (first 5): {test_pred_df.index[:5].tolist()}")
    print(f"Test prediction range (first 5 spots, first gene): {test_pred_df.iloc[:5, 0].values}")

    return test_pred_df, test_spots


def gt_load(test_spots, test_slide, st_path):
    """
    Ground truth 데이터 로드
    
    Parameters:
    -----------
    test_spots: test spot 이름 리스트
    test_slide: test slide 이름
    st_path: ST 데이터 경로
    
    Returns:
    --------
    test_gt_log: DataFrame (spot x genes) - Log2 transformed ground truth
    """
    fpath = os.path.abspath(os.path.normpath(os.path.join(st_path, test_slide + ".h5ad")))
    test_adata = anndata.read_h5ad(fpath)
    X_gt = _to_array(test_adata.X)

    # 인덱스를 test_spots와 일치시킴 (test_pred_df와 동일한 인덱스 사용)
    if len(test_spots) != test_adata.shape[0]:
        print(f"Warning: test_spots count ({len(test_spots)}) != adata spots ({test_adata.shape[0]}).")
        print(f"  test_spots (first 5): {test_spots[:5]}")
        print(f"  adata obs_names (first 5): {test_adata.obs_names[:5].tolist()}")

    # test_spots의 인덱스를 그대로 사용하여 test_pred_df와 일치시킴
    test_gt_df = pd.DataFrame(
        X_gt,
        columns=test_adata.var_names,
        index=test_spots  # test_pred_df와 동일한 인덱스 사용
    )

    # Log transform
    test_gt_log = np.log2(test_gt_df + 1)

    return test_gt_log
