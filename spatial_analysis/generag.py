# -*- coding: utf-8 -*-
"""
GeneRAG Module

Sparse coding 기반 GeneRAG 함수들을 제공합니다.
회귀(Lasso/Ridge/ElasticNet 등)는 모두 scikit-learn 기반으로 수행합니다.
"""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data_loading import load_selected_genes
from .bank_utils import (
    select_high_variable_genes,
    prepare_bank_data,
    get_gene_indices
)


def _load_slide_embedding(embed_dir, slide_id, file_suffix="_uni_aug.pt"):
    """단일 slide의 .pt 임베딩 로드. (n_spots, dim) 또는 (n_spots, n_aug, dim) -> (n_spots, dim) 평균."""
    fpath = os.path.join(embed_dir, slide_id + file_suffix)
    if not os.path.isfile(fpath):
        return None
    emb = torch.load(fpath, map_location="cpu")
    if hasattr(emb, "cpu"):
        emb = emb.cpu().numpy()
    else:
        emb = np.asarray(emb)
    if emb.ndim == 3:
        emb = np.mean(emb, axis=1)
    return emb


def load_bank_embeddings(embed_dir, bank_spot_names, file_suffix="_uni_aug.pt"):
    """Bank spot 순서와 동일한 순서로 임베딩 (N_spots, emb_dim) 반환."""
    spot_meta = []
    for name in bank_spot_names:
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            spot_meta.append((parts[0], int(parts[1])))
        else:
            return None
    slides = list(dict.fromkeys(s[0] for s in spot_meta))
    loaded = {}
    for slide_id in slides:
        emb = _load_slide_embedding(embed_dir, slide_id, file_suffix)
        if emb is None:
            return None
        loaded[slide_id] = emb
    out = np.zeros((len(spot_meta), loaded[slides[0]].shape[1]), dtype=np.float64)
    for i, (sid, idx) in enumerate(spot_meta):
        out[i] = loaded[sid][idx]
    return out


def load_test_embeddings(embed_dir, test_slide, test_spots, file_suffix="_uni_aug.pt"):
    """Test slide 임베딩 로드 후 test_spots 순서대로 (n_test_spots, emb_dim) 반환."""
    emb = _load_slide_embedding(embed_dir, test_slide, file_suffix)
    if emb is None:
        return None
    indices = []
    for name in test_spots:
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            indices.append(int(parts[1]))
        else:
            try:
                indices.append(int(name))
            except ValueError:
                indices.append(len(indices))
    if max(indices) >= emb.shape[0]:
        return None
    return emb[np.array(indices)]


def generag_single_spot_sparse_coding_v2(test_spot_pred, bank_full, gene_indices, valid_genes, 
                                        optimization_method='lasso', optimization_params=None,
                                        bank_embeddings=None, test_spot_embedding=None,
                                        embedding_ratio=0.0, device='cpu'):
    """
    단일 spot에 대해 sparse coding GeneRAG 수행 (다양한 최적화 방법 지원)
    
    embedding_ratio: float in [0, 1]. 유전자 vs 임베딩 비율. gene_weight = 1 - embedding_ratio, embedding_weight = embedding_ratio.
    - 0: 유전자만 사용. 1: 임베딩만 사용. 0.5: 동일 비율.
    목적: (1 - embedding_ratio) * || D@a - y ||^2 + embedding_ratio * || E.T@a - e ||^2
    """
    embedding_ratio = float(np.clip(embedding_ratio, 0.0, 1.0))
    gene_weight = 1.0 - embedding_ratio
    embedding_weight = embedding_ratio

    use_gene = gene_weight > 0
    use_emb = (embedding_weight > 0 and bank_embeddings is not None and test_spot_embedding is not None)
    if not use_gene and not use_emb:
        if embedding_ratio >= 1.0:
            raise ValueError(
                "embedding_ratio=1 (embedding only) but bank_embeddings or test_spot_embedding is missing. "
                "Pass embedding_dir and test_slide to sparse_coding_generag_v2 and ensure .pt files exist."
            )
        raise ValueError(
            "embedding_ratio=0: genes only, 1: embedding only. "
            "For embedding, need embedding_ratio>0 and bank_embeddings, test_spot_embedding."
        )

    D_sub = None
    target_vec = None

    if use_gene:
        if isinstance(test_spot_pred, pd.Series):
            target_300 = test_spot_pred[valid_genes].values
        else:
            target_300 = np.array([test_spot_pred[valid_genes.index(g)] if g in valid_genes else np.nan 
                                   for g in valid_genes])
        valid_mask = ~np.isnan(target_300)
        if valid_mask.sum() < 3:
            raise ValueError(f"Too few valid gene values: {valid_mask.sum()}")
        target_300_valid = target_300[valid_mask]
        gene_indices_valid = gene_indices[valid_mask]
        D_gene = bank_full[gene_indices_valid, :]
        scale_g = np.sqrt(gene_weight)
        D_sub = scale_g * D_gene
        target_vec = scale_g * target_300_valid

    if use_emb:
        E = np.asarray(bank_embeddings)
        e = np.asarray(test_spot_embedding).ravel()
        scale_e = np.sqrt(embedding_weight)
        D_emb = scale_e * E.T
        b_emb = scale_e * e
        if D_sub is not None:
            D_sub = np.vstack([D_sub, D_emb])
            target_vec = np.concatenate([target_vec, b_emb])
        else:
            D_sub = D_emb
            target_vec = b_emb

    # 최적화 파라미터 기본값 설정 (device 인자는 API 호환용으로 받되 사용하지 않음)
    if optimization_params is None:
        optimization_params = {}

    # scikit-learn 기반 회귀
    if optimization_method == 'lasso':
        from sklearn.linear_model import Lasso
        alpha = optimization_params.get('alpha', 0.01)
        positive = optimization_params.get('positive', True)
        max_iter = optimization_params.get('max_iter', 2000)
        clf = Lasso(alpha=alpha, fit_intercept=False, positive=positive, max_iter=max_iter)
        clf.fit(D_sub, target_vec)
        alpha = clf.coef_

    elif optimization_method == 'ridge':
        from sklearn.linear_model import Ridge
        alpha = optimization_params.get('alpha', 1.0)
        clf = Ridge(alpha=alpha, fit_intercept=False)
        clf.fit(D_sub, target_vec)
        alpha = clf.coef_

    elif optimization_method == 'elasticnet':
        from sklearn.linear_model import ElasticNet
        alpha = optimization_params.get('alpha', 0.01)
        l1_ratio = optimization_params.get('l1_ratio', 0.5)
        positive = optimization_params.get('positive', True)
        max_iter = optimization_params.get('max_iter', 2000)
        clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False,
                         positive=positive, max_iter=max_iter)
        clf.fit(D_sub, target_vec)
        alpha = clf.coef_

    elif optimization_method == 'omp':
        from sklearn.linear_model import OrthogonalMatchingPursuit
        n_nonzero_coefs = optimization_params.get('n_nonzero_coefs', None)
        tol = optimization_params.get('tol', None)
        clf = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, tol=tol,
                                        fit_intercept=False)
        clf.fit(D_sub, target_vec)
        alpha = clf.coef_

    elif optimization_method == 'nnls':
        from scipy.optimize import nnls
        alpha, _ = nnls(D_sub, target_vec)

    elif optimization_method == 'lassolars':
        from sklearn.linear_model import LassoLars
        alpha = optimization_params.get('alpha', 0.01)
        positive = optimization_params.get('positive', True)
        clf = LassoLars(alpha=alpha, fit_intercept=False, positive=positive)
        clf.fit(D_sub, target_vec)
        alpha = clf.coef_

    elif optimization_method == 'bayesian_ridge':
        from sklearn.linear_model import BayesianRidge
        alpha_1 = optimization_params.get('alpha_1', 1e-6)
        alpha_2 = optimization_params.get('alpha_2', 1e-6)
        lambda_1 = optimization_params.get('lambda_1', 1e-6)
        lambda_2 = optimization_params.get('lambda_2', 1e-6)
        clf = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1,
                            lambda_2=lambda_2, fit_intercept=False)
        clf.fit(D_sub, target_vec)
        alpha = clf.coef_

    else:
        raise ValueError(f"Unsupported optimization_method: {optimization_method}. "
                        f"Use one of: lasso, ridge, elasticnet, omp, nnls, lassolars, bayesian_ridge")
    
    # Reconstruction (전체 복원)
    reconstructed_full = bank_full @ alpha
    arr = np.array(alpha)
    sparsity = np.count_nonzero(arr)
    
    return reconstructed_full, sparsity


def sparse_coding_generag_v2(test_pred_df, bank_all_df, selected_genes, 
                                optimization_method='lasso', optimization_params=None, 
                                n_high_var_genes=10000,
                                embedding_dir=None, embedding_suffix="_uni_aug.pt",
                                embedding_ratio=0.0, test_slide=None, device='cuda'):
    """
    모든 test spot에 대해 sparse coding 기반 GeneRAG 수행.
    embedding_ratio in [0, 1]: gene_weight = 1 - embedding_ratio, embedding_weight = embedding_ratio.
    - 0: 유전자만 사용 (bank_embeddings 불필요).
    - 1: 임베딩만 사용 (embedding_dir, test_slide 필요).
    - (0, 1): 유전자 + 임베딩 혼합.
    device: API 호환용으로 받으며, 내부에서는 사용하지 않음 (회귀는 scikit-learn 기준).
    """
    embedding_ratio = float(np.clip(embedding_ratio, 0.0, 1.0))
    print("Sparse coding GeneRAG start (High Variable Genes)...")
    print(f"method: {optimization_method}, embedding_ratio: {embedding_ratio} (scikit-learn)")
    
    # 1. High variable genes 선택
    print("  - Selecting high variable genes...")

    high_var_genes = select_high_variable_genes(bank_all_df, n_genes=n_high_var_genes)
    
    # 2. Bank 데이터 준비 (high variable genes만 필터링)
    print("  - Preparing bank data...")
    bank_full, bank_gene_names = prepare_bank_data(bank_all_df, high_var_genes=high_var_genes)
    print(f"    Bank shape: {bank_full.shape} (genes x spots)")
    
    # 3. 유전자 인덱스 매핑 생성
    print("  - Building gene index mapping...")
    gene_indices, valid_genes = get_gene_indices(selected_genes, bank_gene_names)
    print(f"    Common genes: {len(valid_genes)}")
    
    test_spots = test_pred_df.index.tolist()
    bank_embeddings = None
    test_embeddings = None
    if embedding_ratio > 0 and embedding_dir and os.path.isdir(embedding_dir) and test_slide:
        bank_embeddings = load_bank_embeddings(
            embedding_dir, bank_all_df.index.tolist(), file_suffix=embedding_suffix
        )
        test_embeddings = load_test_embeddings(
            embedding_dir, test_slide, test_spots, file_suffix=embedding_suffix
        )
        if bank_embeddings is not None and test_embeddings is not None:
            print(f"  - Using embedding: dir={embedding_dir}, embedding_ratio={embedding_ratio}")
        else:
            bank_embeddings = None
            test_embeddings = None
            if embedding_ratio > 0:
                print("  Warning: embedding load failed, using genes only (embedding_ratio ignored)")
    # embedding_ratio=1(임베딩만)인데 임베딩이 없으면 조기 에러
    if embedding_ratio >= 1.0 and (bank_embeddings is None or test_embeddings is None):
        raise ValueError(
            "embedding_ratio=1 (embedding only) requires embedding_dir and test_slide with bank/test .pt files. "
            "embedding_dir=%r, test_slide=%r. Expect files like %s in the directory."
            % (embedding_dir, test_slide, "<slide_id>" + embedding_suffix)
        )
    
    # 4. 배치 경로: lasso/ridge/elasticnet 이고 동일 valid_mask일 때 sklearn 배치 fit 사용
    use_batch = (optimization_method in ('lasso', 'ridge', 'elasticnet'))
    if use_batch and embedding_ratio < 1.0:
        gene_vals = test_pred_df[valid_genes] if all(g in test_pred_df.columns for g in valid_genes) else None
        if gene_vals is not None:
            has_nan = gene_vals.isna().any(axis=1)
            use_batch = (has_nan.sum() == 0)
    
    generag_list = []
    total_sparsity = 0
    kw = {"embedding_ratio": embedding_ratio}
    if bank_embeddings is not None and test_embeddings is not None:
        kw["bank_embeddings"] = bank_embeddings
    
    if use_batch:
        # 배치 sklearn: D_sub (M,N), B_mat (n_spots,M) -> fit(D_sub.T, B_mat.T) -> coef_ (n_spots,N)
        print("  - Batch GeneRAG (scikit-learn)...")
        gene_weight = 1.0 - embedding_ratio
        embedding_weight = embedding_ratio
        use_gene = gene_weight > 0
        use_emb = (embedding_weight > 0 and bank_embeddings is not None and test_embeddings is not None)
        D_sub = None
        target_rows = None
        if use_gene:
            target_300 = test_pred_df[valid_genes].values.astype(np.float64)
            valid_mask = ~np.isnan(target_300)
            if valid_mask.ndim == 2:
                valid_mask = valid_mask[0]
            gene_indices_valid = gene_indices[valid_mask]
            D_gene = bank_full[gene_indices_valid, :]
            scale_g = np.sqrt(gene_weight)
            D_sub = scale_g * D_gene
            tg = target_300[:, valid_mask] if target_300.ndim == 2 else target_300[valid_mask]
            target_rows = scale_g * (tg.reshape(1, -1) if tg.ndim == 1 else tg)
        if use_emb:
            E = np.asarray(bank_embeddings)
            scale_e = np.sqrt(embedding_weight)
            D_emb = scale_e * E.T
            b_emb = scale_e * np.asarray(test_embeddings)
            if D_sub is not None:
                D_sub = np.vstack([D_sub, D_emb])
                target_rows = np.hstack([target_rows, b_emb])
            else:
                D_sub = D_emb
                target_rows = b_emb
        if target_rows.ndim == 1:
            target_rows = target_rows.reshape(1, -1)
        B_mat = target_rows
        opt_params = optimization_params or {}
        # sklearn multi-output: X = D_sub (M, N), y = B_mat.T (M, n_spots) -> coef_ (n_spots, N)
        X_batch = D_sub
        y_batch = B_mat.T
        if optimization_method == 'lasso':
            from sklearn.linear_model import Lasso
            model = Lasso(
                alpha=opt_params.get('alpha', 0.01),
                fit_intercept=False,
                positive=opt_params.get('positive', True),
                max_iter=opt_params.get('max_iter', 2000)
            )
        elif optimization_method == 'ridge':
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=opt_params.get('alpha', 1.0), fit_intercept=False)
        else:
            from sklearn.linear_model import ElasticNet
            model = ElasticNet(
                alpha=opt_params.get('alpha', 0.01),
                l1_ratio=opt_params.get('l1_ratio', 0.5),
                fit_intercept=False,
                positive=opt_params.get('positive', True),
                max_iter=opt_params.get('max_iter', 2000)
            )
        model.fit(X_batch, y_batch)
        alphas = np.asarray(model.coef_).astype(np.float64)
        if alphas.ndim == 1:
            alphas = alphas.reshape(1, -1)
        reconstructed_all = (bank_full @ alphas.T).T
        for i, spot_name in enumerate(tqdm(test_spots, desc="    GeneRAG spots (batch)")):
            total_sparsity += np.count_nonzero(alphas[i])
            generag_list.append(pd.Series(reconstructed_all[i], index=bank_gene_names, name=spot_name))
    else:
        # 단일 spot 루프
        print("  - GeneRAG each spot...")
        for spot_idx, spot_name in enumerate(tqdm(test_spots, desc="    GeneRAG spots")):
            test_spot_pred = test_pred_df.loc[spot_name]
            if bank_embeddings is not None and test_embeddings is not None:
                kw["test_spot_embedding"] = test_embeddings[spot_idx]
            reconstructed, sparsity = generag_single_spot_sparse_coding_v2(
                test_spot_pred, bank_full, gene_indices, valid_genes, 
                optimization_method=optimization_method, 
                optimization_params=optimization_params,
                device=device,
                **kw
            )
            total_sparsity += sparsity
            generag_series = pd.Series(reconstructed, index=bank_gene_names, name=spot_name)
            generag_list.append(generag_series)
    
    total_sparsity = total_sparsity / len(test_spots)
    
    # 5. 결과 통합
    print("  - Merging results...")
    generag_df = pd.DataFrame(generag_list)
    print(f"    GeneRAG done: {generag_df.shape} (spots x {n_high_var_genes} high variable genes)")
    print(f"    Mean sparsity: {total_sparsity:.2f}")
    
    return generag_df, total_sparsity
