# Spatial Transcriptomics Optimization (exp_opt_v2)

## 프로젝트 소개

- **이름**: Spatial Transcriptomics Optimization (exp_opt_v2)
- **목적**: UNI, CONCH, Exaone 등 모델의 저해상도 예측 결과에 대해 **GeneRAG(sparse coding)** 최적화 실험을 자동으로 수행하고, 하이퍼파라미터(alpha, embedding_ratio 등) 탐색 결과를 CSV로 저장합니다.
- **핵심 흐름**: `init_pred_pt` 디렉터리의 `.pt` 예측 파일 스캔 → 모델/유전자 리스트별로 Bank 데이터 로드 → 테스트 슬라이드 예측·GT 로드 → `run_optimization_experiment` 실행 → 결과 CSV 저장

## 프로젝트 구조

| 경로 | 설명 |
|------|------|
| `main.py` | 진입점. lr_pred_pt 스캔, 모델별 최적화 실험 실행, 결과 저장 |
| `spatial_analysis/` | 분석·평가 패키지 |
| `spatial_analysis/config.py` | `search_space` (lasso, elasticnet, ridge 파라미터 그리드) |
| `spatial_analysis/data_loading.py` | 유전자 리스트, Bank, 테스트 예측·GT 로드 |
| `spatial_analysis/generag.py` | Sparse coding GeneRAG (Lasso/Ridge/ElasticNet, 선택적 UNI 임베딩) |
| `spatial_analysis/bank_utils.py` | High variable genes, bank 행렬 준비, 유전자 인덱스 |
| `spatial_analysis/evaluation.py` | 단일/일괄 실험 실행, PCC/MSE/RVD/sparsity 평가 |
| `selected_gene/` | 유전자 리스트 `.txt` (예: selected_morph_top50_gene_list.txt) |

## 데이터 요구사항

- **경로 설정** (`main.py` 상단):
  - `data_path`: HER2ST processed_data 디렉터리 (co-expression 유전자 리스트, 임베딩 등)
  - `st_path`: 슬라이드별 `.h5ad` 경로 (예: SPA119.h5ad ~ SPA148.h5ad)
  - `lr_pred_pt_dir` (기본 `./init_pred_pt`): 예측 `.pt` 파일 디렉터리
  - `selected_gene_dir` (기본 `./selected_gene`): 유전자 리스트 `.txt` 디렉터리
- **예측 파일 네이밍**: `generated_samples_lr_{MODEL}_{gene_basename}_20sample.pt`
  - `selected_gene` 내 `{gene_basename}.txt`가 존재해야 해당 조합이 실험 대상에 포함됩니다.
- **임베딩** (embedding_ratio > 0 사용 시): `data_path` 내 `1spot_uni_ebd_aug` 등, 슬라이드별 `{slide_id}_uni_aug.pt` 형식

## 설정 요약 (main.py)

- **테스트 슬라이드**: `test_slide = "SPA148"`, **학습(Bank) 슬라이드**: SPA119 ~ SPA153 (테스트 제외)
- **실험 옵션**: `n_high_var_genes`, `calibration_method` (예: `"log1p"`), `n_jobs` (병렬 GPU 프로세스 수), `embedding_dir`

## 실행 방법

```bash
cd /mnt/nas1/physical_ai/hyeongsub.kim/proj/Stem/exp_opt_v2
python main.py
```

`init_pred_pt`에 위 네이밍 규칙에 맞는 `.pt` 파일과 `selected_gene`에 대응하는 `.txt`가 있어야 하며, `st_path`와 `data_path`가 실제 데이터를 가리켜야 합니다.

## 검색 공간 (search_space) 요약

- **lasso**: alpha, fit_intercept, embedding_ratio (0~1)
- **elasticnet**: alpha, l1_ratio, embedding_ratio
- **ridge**: alpha, solver (embedding 미사용)

`embedding_ratio`: 0=유전자만, 1=임베딩만, 0.5=동일 비율 (gene + embedding 결합 목적함수).

## 출력 결과

- **저장 위치**: `save_path` (기본 `./results`)
- **파일명**: `{exp_name}_optimization_experiment_results_{model_name}_{gene_basename}.csv`
- **컬럼**: optimization_method, alpha, embedding_ratio 등 파라미터 + pcc_10, pcc_50, pcc_300, mse, rvd, sparsity 등

## 의존성

- `numpy`, `pandas`, `torch`, `anndata`, `scipy`, `scikit-learn`, `tqdm`
- requirements.txt는 프로젝트에 포함되어 있지 않으며, 필요 시 별도로 관리하면 됩니다.
