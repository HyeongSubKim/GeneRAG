# -*- coding: utf-8 -*-
"""
Evaluation Module

Imputation 결과 평가 및 분석 함수들을 제공합니다.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from scipy import stats
from datetime import datetime, timedelta
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from .generag import sparse_coding_generag_v2


def calculate_spot_correlation(generag_df, ground_truth_df, spot_name):
    """Spot별 전체 correlation 계산"""
    # 인덱스 확인
    if spot_name not in generag_df.index:
        return np.nan
    if spot_name not in ground_truth_df.index:
        return np.nan
    
    common_genes = list(set(generag_df.columns) & set(ground_truth_df.columns))
    if len(common_genes) == 0:
        return np.nan
    
    try:
        generag_vals = generag_df.loc[spot_name, common_genes].values
        gt_vals = ground_truth_df.loc[spot_name, common_genes].values
        
        # NaN 처리
        mask = ~(np.isnan(generag_vals) | np.isnan(gt_vals))
        if mask.sum() < 3:
            return np.nan
        
        # 분산이 0인 경우 처리
        if np.std(generag_vals[mask]) == 0 or np.std(gt_vals[mask]) == 0:
            return np.nan
        
        corr = np.corrcoef(generag_vals[mask], gt_vals[mask])[0, 1]
        return corr if not np.isnan(corr) else np.nan
    except Exception as e:
        return np.nan


def calculate_gene_correlation(generag_df, ground_truth_df, gene_name):
    """유전자별 correlation 계산"""
    common_spots = list(set(generag_df.index) & set(ground_truth_df.index))
    
    if gene_name not in generag_df.columns or gene_name not in ground_truth_df.columns:
        return np.nan
    
    generag_vals = generag_df.loc[common_spots, gene_name].values
    gt_vals = ground_truth_df.loc[common_spots, gene_name].values
    
    mask = ~(np.isnan(generag_vals) | np.isnan(gt_vals))
    if mask.sum() < 3:
        return np.nan
    
    corr = np.corrcoef(generag_vals[mask], gt_vals[mask])[0, 1]
    return corr if not np.isnan(corr) else 0.0


def gene_correlation_analysis(test_pred_df, test_gt_log, selected_genes):
    """
    유전자별 correlation 분석
    
    Parameters:
    -----------
    test_pred_df: DataFrame - Test 예측 데이터
    test_gt_log: DataFrame - Ground truth 데이터
    selected_genes: list - 분석할 유전자 리스트
    
    Returns:
    --------
    gene_300_correlations: numpy array - 유전자별 correlation 값들
    """
    gene_correlations = []
    gene_corr_list = []

    for gene in tqdm(selected_genes):
        corr = calculate_gene_correlation(test_pred_df, test_gt_log, gene)
        if not np.isnan(corr):
            gene_correlations.append(corr)
            gene_corr_list.append({'Gene': gene, 'Correlation': corr})

    gene_correlations = np.array(gene_correlations)
    gene_corr_df = pd.DataFrame(gene_corr_list).sort_values('Correlation', ascending=False)
    df_selected = gene_corr_df[gene_corr_df["Gene"].isin(selected_genes)]
    gene_300_correlations = np.array(list(df_selected['Correlation']))
    
    return gene_300_correlations


def evaluate_generag_result(generag_df, test_gt_log, calibration_method=None):
    """
    GeneRAG 결과를 평가하는 함수
    
    Parameters:
    -----------
    generag_df: DataFrame (spot x genes) - GeneRAG 결과
    test_gt_log: DataFrame (spot x genes) - Ground truth 데이터
    calibration_method: str - Calibration 방법 ('log2', 'log1p', 'quantile', 'zscore', None)
    
    Returns:
    --------
    results: dict - 평가 결과 (correlation metrics, MSE, MAE, RVD)
    """
    # Calibration 적용
    if calibration_method == 'log2':
        generag_df_calibrated = np.log2(generag_df + 1)
    elif calibration_method == 'log1p':
        generag_df_calibrated = np.log1p(generag_df)
    elif calibration_method == 'quantile':
        generag_df_calibrated = generag_df.copy()
        for col in generag_df.columns:
            if generag_df[col].notna().sum() > 0:
                generag_df_calibrated[col] = stats.norm.ppf(
                    stats.rankdata(generag_df[col].fillna(0), method='average') / (len(generag_df) + 1)
                )
    elif calibration_method == 'zscore':
        generag_df_calibrated = generag_df.copy()
        for col in generag_df.columns:
            col_mean = generag_df[col].mean()
            col_std = generag_df[col].std()
            if col_std > 0:
                generag_df_calibrated[col] = (generag_df[col] - col_mean) / col_std
    else:
        generag_df_calibrated = generag_df.copy()
    
    # 1. Correlation 계산
    total_genes = list(generag_df_calibrated.columns)
    gene_correlations = gene_correlation_analysis(generag_df_calibrated, test_gt_log, total_genes)
    
    # 상위 N개 유전자의 평균 correlation
    sorted_correlations = sorted(gene_correlations)[::-1]
    pcc_10 = np.mean(sorted_correlations[:10]) if len(sorted_correlations) >= 10 else np.nan
    pcc_50 = np.mean(sorted_correlations[:50]) if len(sorted_correlations) >= 50 else np.nan
    pcc_300 = np.mean(sorted_correlations[:300]) if len(sorted_correlations) >= 300 else np.nan
    pcc_1000 = np.mean(sorted_correlations[:1000]) if len(sorted_correlations) >= 1000 else np.nan
    pcc_2000 = np.mean(sorted_correlations[:2000]) if len(sorted_correlations) >= 2000 else np.nan
    pcc_3000 = np.mean(sorted_correlations[:3000]) if len(sorted_correlations) >= 3000 else np.nan
    pcc_5000 = np.mean(sorted_correlations[:5000]) if len(sorted_correlations) >= 5000 else np.nan
    pcc_10000 = np.mean(sorted_correlations[:10000]) if len(sorted_correlations) >= 10000 else np.nan
    
    # 2. MSE, MAE 계산 (PCC top 300 유전자 사용)
    # PCC top 300 유전자 선택
    gene_corr_pairs = list(zip(total_genes, gene_correlations))
    sorted_gene_corr = sorted(gene_corr_pairs, key=lambda x: x[1], reverse=True)
    pcc_top300_genes = [gene for gene, corr in sorted_gene_corr[:300]]
    
    # 공통 유전자와 공통 spot 찾기
    common_genes = pcc_top300_genes
    common_spots = list(set(generag_df_calibrated.index) & set(test_gt_log.index))
    
    if len(common_genes) > 0 and len(common_spots) > 0:
        # 공통 유전자와 공통 spot에 대해 데이터 정렬
        generag_aligned = generag_df_calibrated.loc[common_spots, common_genes].values
        gt_aligned = test_gt_log.loc[common_spots, common_genes].values
        
        # NaN 값 처리
        mask = ~(np.isnan(generag_aligned) | np.isnan(gt_aligned))
        valid_values = mask.sum()
        
        if valid_values > 0:
            diff = gt_aligned - generag_aligned
            diff_valid = diff[mask]
            mse = np.mean(diff_valid**2)
            mae = np.mean(np.abs(diff_valid))
        else:
            mse = np.nan
            mae = np.nan
        
        # 3. RVD 계산
        generag_var = np.nanvar(generag_aligned, axis=0)
        gt_var = np.nanvar(gt_aligned, axis=0)
        
        valid_var_mask = np.isfinite(generag_var) & np.isfinite(gt_var) & (gt_var > 0)
        
        if valid_var_mask.sum() > 0:
            rvd_values = (generag_var[valid_var_mask] - gt_var[valid_var_mask])**2 / gt_var[valid_var_mask]**2
            rvd = np.mean(rvd_values)
        else:
            rvd = np.nan
    else:
        mse = np.nan
        mae = np.nan
        rvd = np.nan
    
    return {
        'pcc_10': pcc_10,
        'pcc_50': pcc_50,
        'pcc_300': pcc_300,
        'pcc_1000': pcc_1000,
        'pcc_2000': pcc_2000,
        'pcc_3000': pcc_3000,
        'pcc_5000': pcc_5000,
        'pcc_10000': pcc_10000,
        'mse': mse,
        'mae': mae,
        'rvd': rvd
    }


def _run_single_experiment(args):
    """
    단일 실험을 실행하는 독립적인 함수 (멀티 GPU 프로세스용)
    
    Parameters:
    -----------
    args: tuple - (experiment_id, optimization_method, params, test_pred_df, bank_all_df,
                   selected_genes, test_gt_log, n_high_var_genes, calibration_method,
                   embedding_dir, test_slide[, device_id])
    
    Returns:
    --------
    result: dict - 실험 결과
    """
    # args 길이에 따라 device_id 추출 (12개면 device_id 포함, 11개면 없음)
    if len(args) == 12:
        (experiment_id, optimization_method, params, test_pred_df, bank_all_df,
         selected_genes, test_gt_log, n_high_var_genes, calibration_method,
         embedding_dir, test_slide, device_id) = args
    elif len(args) == 11:
        (experiment_id, optimization_method, params, test_pred_df, bank_all_df,
         selected_genes, test_gt_log, n_high_var_genes, calibration_method,
         embedding_dir, test_slide) = args
        device_id = None
    else:
        raise ValueError(f"_run_single_experiment: unexpected args length {len(args)}, expected 11 or 12")

    # embedding_ratio는 GeneRAG 전용 (0~1, gene_weight + embedding_weight = 1); optimization_params에서 제외
    optimization_params = {k: v for k, v in params.items() if k not in ('embedding_ratio', 'embedding_weight')}
    embedding_ratio = params.get('embedding_ratio')
    if embedding_ratio is None:
        embedding_ratio = params.get('embedding_weight', 0)
    embedding_ratio = float(np.clip(embedding_ratio, 0.0, 1.0))
    
    device = f'cuda:{device_id}' if (device_id is not None and torch.cuda.is_available() and device_id < torch.cuda.device_count()) else 'cuda'
    
    try:
        # GeneRAG 수행 (지정 GPU 또는 기본 cuda)
        generag_df, total_sparsity = sparse_coding_generag_v2(
            test_pred_df=test_pred_df,
            bank_all_df=bank_all_df,
            selected_genes=selected_genes,
            optimization_method=optimization_method,
            optimization_params=optimization_params,
            n_high_var_genes=n_high_var_genes,
            embedding_dir=embedding_dir,
            embedding_ratio=embedding_ratio,
            test_slide=test_slide,
            device=device
        )
        
        # 결과 평가
        eval_results = evaluate_generag_result(
            generag_df, test_gt_log, calibration_method=calibration_method
        )
        
        # 결과 저장 (embedding_ratio는 0~1로 통일해 저장)
        result_row = {
            'experiment_id': experiment_id,
            'optimization_method': optimization_method,
            'sparsity': total_sparsity,
            **params,
            'embedding_ratio': embedding_ratio,
            **eval_results
        }
        
        return result_row
        
    except Exception as e:
        # 오류 발생 시에도 결과에 기록
        result_row = {
            'experiment_id': experiment_id,
            'optimization_method': optimization_method,
            'sparsity': np.nan,
            **params,
            'embedding_ratio': embedding_ratio,
            'pcc_10': np.nan,
            'pcc_50': np.nan,
            'pcc_300': np.nan,
            'pcc_1000': np.nan,
            'pcc_2000': np.nan,
            'pcc_3000': np.nan,
            'pcc_5000': np.nan,
            'pcc_10000': np.nan,
            'mse': np.nan,
            'mae': np.nan,
            'rvd': np.nan,
            'error': str(e)
        }
        return result_row


def run_optimization_experiment(test_pred_df, bank_all_df, selected_genes, test_gt_log, 
                                search_space, n_high_var_genes=10000, calibration_method=None,
                                output_path=None, save_intermediate=True, intermediate_save_interval=1,
                                n_jobs=1, embedding_dir=None, test_slide=None):
    """
    search_space에 정의된 각 최적화 방법과 파라미터 조합에 대해 실험 수행
    
    Parameters:
    -----------
    test_pred_df: DataFrame (spot x selected_genes) - Test 예측 데이터
    bank_all_df: DataFrame (spot x all_genes) - Bank 전체 데이터
    selected_genes: list - 선택된 유전자 리스트
    test_gt_log: DataFrame (spot x genes) - Ground truth 데이터
    search_space: dict - 최적화 방법별 파라미터 조합
    n_high_var_genes: int - 사용할 high variable genes 수 (기본값: 10000)
    calibration_method: str - Calibration 방법 (기본값: None)
    output_path: str - 결과 CSV 저장 경로 (기본값: None)
    save_intermediate: bool - 중간 결과 저장 여부 (기본값: True)
    intermediate_save_interval: int - 중간 결과 저장 주기 (최적화 방법별, 기본값: 1)
    n_jobs: int - 병렬 실행할 GPU 프로세스 수 (기본값: 1, 순차 실행). >1 이면 ProcessPoolExecutor로 멀티 GPU 분배
    embedding_dir: str or None - L2 embedding term용 임베딩 디렉토리 (기본값: None)
    test_slide: str or None - test slide ID, embedding_dir 사용 시 필요 (기본값: None)
    
    Returns:
    --------
    res_df: DataFrame - 실험 결과
    """
    # 전체 실험 수 계산
    total_experiments = 0
    for param_space in search_space.values():
        if len(param_space) == 0:
            total_experiments += 1
        else:
            param_values = [param_space[key] for key in param_space.keys()]
            total_experiments += len(list(product(*param_values)))
    
    print(f"\n{'='*70}", flush=True)
    print(f"Optimization experiment start", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Total experiments: {total_experiments}", flush=True)
    print(f"Optimization methods: {len(search_space)}", flush=True)
    n_gpus = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    print(f"Parallel: {'Multi-GPU (ProcessPool)' if n_jobs > 1 else 'Off'} (n_jobs={n_jobs}, GPUs: {min(n_jobs, n_gpus)})", flush=True)
    print(f"Save intermediate: {'On' if save_intermediate else 'Off'}", flush=True)
    if save_intermediate:
        print(f"Intermediate save: every method completion", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    # 시작 시간 기록
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)
    
    # 진행 상황 추적을 위한 변수
    completed_count = 0
    res_list = []
    
    # 모든 실험 작업 생성
    experiment_tasks = []
    experiment_id = 0
    
    for optimization_method, param_space in search_space.items():
        # 파라미터 조합 생성
        if len(param_space) == 0:
            param_combinations = [{}]
        else:
            param_keys = list(param_space.keys())
            param_values = [param_space[key] for key in param_keys]
            param_combinations = [dict(zip(param_keys, combo)) for combo in product(*param_values)]
        
        for params in param_combinations:
            experiment_id += 1
            experiment_tasks.append((
                experiment_id,
                optimization_method,
                params,
                test_pred_df,
                bank_all_df,
                selected_genes,
                test_gt_log,
                n_high_var_genes,
                calibration_method,
                embedding_dir,
                test_slide
            ))
    
    # 병렬 또는 순차 실행
    if n_jobs > 1:
        # 멀티 GPU: 각 작업에 device_id 할당 (task_idx % n_gpus)
        workers = min(n_jobs, n_gpus)
        tasks_with_device = [(*t, i % n_gpus) for i, t in enumerate(experiment_tasks)]
        print(f"Multi-GPU run start (workers: {workers}, GPU: 0~{n_gpus-1})...", flush=True)
        print("(Worker GeneRAG logs are not shown; progress is printed on each completion.)\n", flush=True)
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # 모든 작업 제출
            future_to_task = {executor.submit(_run_single_experiment, task): task for task in tasks_with_device}
            
            # 완료된 작업 처리
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                experiment_id, optimization_method, params = task[0], task[1], task[2]
                gpu_id = task[11] if len(task) == 12 else None
                
                try:
                    result = future.result()
                    
                    # 결과 추가
                    res_list.append(result)
                    completed_count += 1
                    
                    # 진행 상황 출력
                    overall_progress = (completed_count / total_experiments) * 100
                    elapsed_time = datetime.now() - start_time
                    elapsed_seconds = elapsed_time.total_seconds()
                    
                    if completed_count > 1:
                        avg_time_per_experiment = elapsed_seconds / (completed_count - 1)
                        remaining_experiments = total_experiments - completed_count
                        estimated_remaining_seconds = avg_time_per_experiment * remaining_experiments
                        estimated_end_time = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
                    else:
                        estimated_end_time = None
                    
                    gpu_info = f" GPU {gpu_id}" if gpu_id is not None else ""
                    print(f"[{completed_count}/{total_experiments}] progress: {overall_progress:.1f}% | "
                          f"method: {optimization_method}{gpu_info} | params: {params}", flush=True)
                    print(f"  elapsed: {str(elapsed_time).split('.')[0]}", end="", flush=True)
                    if estimated_end_time:
                        print(f" | ETA: {estimated_end_time.strftime('%H:%M:%S')}", flush=True)
                    else:
                        print(flush=True)
                    
                    if 'error' not in result:
                        print(f"  OK Sparsity={result['sparsity']:.2f}, "
                              f"PCC-300={result['pcc_300']:.4f}, MSE={result['mse']:.4f}", flush=True)
                    else:
                        print(f"  ERROR: {result.get('error', 'Unknown error')}", flush=True)
                    print(flush=True)
                    
                    # 중간 결과 저장 (주기적으로)
                    if save_intermediate and output_path is not None and completed_count % (total_experiments // 10 + 1) == 0:
                        intermediate_df = pd.DataFrame(res_list)
                        # experiment_id로 정렬
                        if 'experiment_id' in intermediate_df.columns:
                            intermediate_df = intermediate_df.sort_values('experiment_id')
                        base_name = os.path.splitext(output_path)[0]
                        ext = os.path.splitext(output_path)[1]
                        intermediate_path = f"{base_name}_intermediate{ext}"
                        intermediate_df.to_csv(intermediate_path, index=False)
                        print(f"Saved intermediate: {intermediate_path} ({completed_count} done)\n", flush=True)
                
                except Exception as e:
                    print(f"Experiment {experiment_id} error: {str(e)}\n", flush=True)
    else:
        # 순차 실행 (기존 방식)
        print("Sequential run start...\n")
        
        for method_idx, (optimization_method, param_space) in enumerate(search_space.items(), 1):
            method_start_time = datetime.now()
            print(f"\n{'='*70}")
            print(f"[{method_idx}/{len(search_space)}] method: {optimization_method}")
            print(f"{'='*70}")
            
            # 파라미터 조합 생성
            if len(param_space) == 0:
                param_combinations = [{}]
            else:
                param_keys = list(param_space.keys())
                param_values = [param_space[key] for key in param_keys]
                param_combinations = [dict(zip(param_keys, combo)) for combo in product(*param_values)]
            
            num_params = len(param_combinations)
            print(f"Param combinations: {num_params}")
            print(f"Start: {method_start_time.strftime('%H:%M:%S')}\n")
            
            for param_idx, params in enumerate(param_combinations, 1):
                completed_count += 1
                
                # 전체 진행률 계산
                overall_progress = (completed_count / total_experiments) * 100
                
                # 경과 시간 계산
                elapsed_time = datetime.now() - start_time
                elapsed_seconds = elapsed_time.total_seconds()
                
                # 예상 완료 시간 계산
                if completed_count > 1:
                    avg_time_per_experiment = elapsed_seconds / (completed_count - 1)
                    remaining_experiments = total_experiments - completed_count
                    estimated_remaining_seconds = avg_time_per_experiment * remaining_experiments
                    estimated_end_time = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
                else:
                    estimated_end_time = None
                
                print(f"\n[{completed_count}/{total_experiments}] progress: {overall_progress:.1f}%")
                print(f"  method: {optimization_method} [{param_idx}/{num_params}]")
                print(f"  params: {params}")
                print(f"  elapsed: {str(elapsed_time).split('.')[0]}")
                if estimated_end_time:
                    print(f"  ETA: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 단일 실험 실행
                task = (completed_count, optimization_method, params, test_pred_df, bank_all_df,
                       selected_genes, test_gt_log, n_high_var_genes, calibration_method,
                       embedding_dir, test_slide)
                result = _run_single_experiment(task)
                res_list.append(result)
                
                if 'error' not in result:
                    experiment_time = datetime.now() - start_time
                    print(f"  OK Sparsity={result['sparsity']:.2f}, "
                          f"PCC-300={result['pcc_300']:.4f}, MSE={result['mse']:.4f}")
                    print(f"  experiment time: {str(experiment_time - elapsed_time).split('.')[0]}")
                else:
                    print(f"  ERROR: {result.get('error', 'Unknown error')}")
            
            # 최적화 방법별 완료 시간
            method_elapsed_time = datetime.now() - method_start_time
            print(f"\n{'='*70}")
            print(f"Done [{method_idx}/{len(search_space)}] {optimization_method}")
            print(f"   elapsed: {str(method_elapsed_time).split('.')[0]}")
            print(f"   end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")
            
            # 중간 결과 저장 (각 최적화 방법 완료 시마다)
            if save_intermediate and output_path is not None and (method_idx % intermediate_save_interval == 0):
                intermediate_df = pd.DataFrame(res_list)
                # experiment_id로 정렬
                if 'experiment_id' in intermediate_df.columns:
                    intermediate_df = intermediate_df.sort_values('experiment_id')
                base_name = os.path.splitext(output_path)[0]
                ext = os.path.splitext(output_path)[1]
                intermediate_path = f"{base_name}_intermediate{ext}"
                intermediate_df.to_csv(intermediate_path, index=False)
                print(f"Saved intermediate: {intermediate_path} ({len(res_list)} done)")
    
    # 전체 실험 완료
    end_time = datetime.now()
    total_elapsed_time = end_time - start_time
    
    # DataFrame으로 변환 및 정렬
    res_df = pd.DataFrame(res_list)
    if 'experiment_id' in res_df.columns:
        res_df = res_df.sort_values('experiment_id')
        res_df = res_df.drop('experiment_id', axis=1)  # experiment_id 컬럼 제거
    
    # 최종 결과 CSV 저장
    if output_path is not None:
        res_df.to_csv(output_path, index=False)
        print(f"\n{'='*70}", flush=True)
        print(f"Results saved: {output_path}", flush=True)
        print(f"{'='*70}", flush=True)
    
    # 최종 요약 출력
    print(f"\n{'='*70}", flush=True)
    print(f"Experiment summary", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Total time: {str(total_elapsed_time).split('.')[0]}", flush=True)
    print(f"Done: {len(res_df)} / {total_experiments} experiments", flush=True)
    print(f"Avg per experiment: {total_elapsed_time.total_seconds() / len(res_df):.2f}s", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    return res_df
