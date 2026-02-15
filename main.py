# -*- coding: utf-8 -*-
"""
Spatial Transcriptomics Optimization - UNI, CONCH, Exaone 결과 기반

main.py와 동일한 최적화 실험을 UNI, CONCH, Exaone 세 모델의 예측 결과에 대해 각각 실행합니다.
evaluation.run_optimization_experiment를 모델별로 한 번씩 호출하고, 결과를 모델별 CSV로 저장합니다.
"""

import os

from spatial_analysis import (
    load_selected_genes,
    load_bank_data,
    infer_test_spots,
    gt_load,
    run_optimization_experiment,
    search_space,
)

# ====================================================
# 설정
# ====================================================
exp_name = '260215_GeneRAG_test'
data_path = "/mnt/nas1/physical_ai/hyeongsub.kim/proj/Stem/hest1k_datasets/her2st/processed_data/"
st_path = "/mnt/nas1/physical_ai/hyeongsub.kim/proj/Stem/hest1k_datasets/her2st/st/"
lr_pred_pt_dir = './init_pred_pt' # 실험 하고자 하는 init_pred_pt 디렉토리 'generated_samples_lr_{model_name}_{seleced_gene}_list_20sample.pt'을 따름
selected_gene_dir = './selected_gene' # 실험 하고자 하는 selected_gene.txt 파일이 있는 디렉토리
self_anchor=False # 지금 실험에서는 고정
if not self_anchor:
    chain_genes_file = selected_gene_dir + "selected_co-expression_gene_list.txt" ########### 이부분 변경하여 실험
    coexp_genes = load_selected_genes(chain_genes_file)
else:
    coexp_genes = None

######################################################

test_slide = "SPA148"
train_slides = ["SPA" + str(i) for i in range(119, 154)]
if test_slide in train_slides:
    train_slides.remove(test_slide)

# UNI, CONCH, Exaone 예측 파일 경로 (.pt, infer_test_spots 형식)
# lr_pred_pt 내 모델들은 selected_gene 파일명과 매핑: generated_samples_lr_{MODEL}_{selected_gene_basename}_20sample.pt
lr_pred_pt_dir = './init_pred_pt'
selected_gene_dir = './selected_gene'

# 실험 공통 파라미터
n_high_var_genes = 10000
calibration_method = "log1p"
save_intermediate = False
intermediate_save_interval = 1
n_jobs = 20

# embedding_ratio > 0 또는 =1 일 때 필수: 임베딩 디렉터리 (슬라이드별 <slide_id>_uni_aug.pt 등)
embedding_dir = os.path.join(data_path, "1spot_uni_ebd_aug")

# lr_pred_pt 내 모든 .pt 파일 스캔: generated_samples_lr_{MODEL}_{gene_basename}_20sample.pt
def _scan_lr_pred_pt_models(lr_pred_pt_dir, selected_gene_dir):
    """lr_pred_pt 디렉토리의 .pt 파일을 스캔하여 (model_name, pred_path, gene_basename) 리스트 반환"""
    import glob
    pattern = os.path.join(lr_pred_pt_dir, "generated_samples_lr_*_*_20sample.pt")
    files = glob.glob(pattern)
    tasks = []
    for fpath in sorted(files):
        basename = os.path.basename(fpath)
        if not basename.startswith("generated_samples_lr_") or not basename.endswith("_20sample.pt"):
            continue
        inner = basename[len("generated_samples_lr_"):-len("_20sample.pt")]
        parts = inner.split("_", 1)
        if len(parts) != 2:
            continue
        model_name, gene_basename = parts[0], parts[1]
        gene_file = os.path.join(selected_gene_dir, gene_basename + ".txt")
        if not os.path.isfile(gene_file):
            continue
        tasks.append((model_name, fpath, gene_basename))
    return tasks

# ====================================================
# lr_pred_pt 전체 모델 자동 실행
# ====================================================

def _log(msg, flush=True):
    print(msg, flush=flush)


if __name__ == "__main__":
    tasks = _scan_lr_pred_pt_models(lr_pred_pt_dir, selected_gene_dir)
    _log("=" * 70)
    _log("lr_pred_pt: multi-model optimization")
    _log("=" * 70)
    _log(f"Test slide: {test_slide}")
    _log(f"Tasks (model, gene_list): {len(tasks)}")
    for m, p, g in tasks:
        _log(f"  - {m} / {g}")
    _log(f"High variable genes: {n_high_var_genes}")
    _log(f"Calibration: {calibration_method}")
    _log(f"n_jobs: {n_jobs}")
    _log(f"embedding_dir: {embedding_dir}")
    _log("=" * 70)

    all_results = {}

    for model_name, pred_path, gene_basename in tasks:
        if not self_anchor:
            selected_genes_file = os.path.join(selected_gene_dir, gene_basename + ".txt")
            selected_genes = load_selected_genes(selected_genes_file)
            anchor_genes = list(set(selected_genes) & set(coexp_genes))
        else:
            anchor_genes = selected_genes

        _log(f"\n{'='*70}")
        _log(f"Model: {model_name} | Gene list: {gene_basename}")
        _log(f"Pred path: {pred_path}")
        _log(f"{'='*70}\n")

        bank_selected_df, bank_all_df = load_bank_data(train_slides, selected_genes, st_path)

        test_pred_df, test_spots = infer_test_spots(
            pred_path, test_slide, selected_genes, st_path
        )
        test_gt_log = gt_load(test_spots, test_slide, st_path)

        run_key = f"{model_name}_{gene_basename}"
        output_path = os.path.join(
            save_path,
            f"{exp_name}_optimization_experiment_results_{run_key}.csv",
        )

        res_df = run_optimization_experiment(
            test_pred_df=test_pred_df,
            bank_all_df=bank_all_df,
            selected_genes=anchor_genes,
            test_gt_log=test_gt_log,
            search_space=search_space,
            n_high_var_genes=n_high_var_genes,
            calibration_method=calibration_method,
            output_path=output_path,
            save_intermediate=save_intermediate,
            intermediate_save_interval=intermediate_save_interval,
            n_jobs=n_jobs,
            embedding_dir=embedding_dir,
            test_slide=test_slide,
        )

        all_results[run_key] = res_df
        _log(f"\nDone: {run_key} ({len(res_df)} experiments) -> {output_path}")

    _log("\n" + "=" * 70)
    _log("All experiments done")
    _log("=" * 70)
    for run_key, res_df in all_results.items():
        _log(f"\n[{run_key}] Summary (mean by optimization_method):")
        print(res_df.groupby("optimization_method")[["pcc_300", "mse", "rvd", "sparsity"]].mean(), flush=True)
