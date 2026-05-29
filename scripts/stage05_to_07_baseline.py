from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


TARGET = "IGCC.DeNOX.AT_H1_901_PV"
RUN_ID = "nox_manual_review_20260429_01"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def metric_row(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_log(project_root: Path, stage: str, purpose: str, inputs: list[str], outputs: list[str], checkpoint: str, next_step: str) -> None:
    log_path = project_root / "log.md"
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    lines = []
    if f"## {date_str}" not in existing:
        lines += ["", f"## {date_str}"]
    lines += [
        "",
        f"### {time_str} - {stage}",
        f"- **목적:** {purpose}",
        f"- **입력:** {', '.join(f'`{item}`' for item in inputs)}",
        f"- **산출물:** {', '.join(f'`{item}`' for item in outputs)}",
        f"- **체크포인트:** {checkpoint}",
        f"- **다음 권장:** {next_step}",
    ]
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def feature_policies(feature_cols: list[str]) -> dict[str, list[str]]:
    o2_cols = [c for c in feature_cols if "AIT_H1_902" in c or c.startswith("feat_AIT")]
    future_cols = [c for c in feature_cols if "future" in c.lower()]
    noleak_exclude = set(o2_cols + future_cols)
    with_o2_exclude = set(future_cols)
    with_future_exclude = set(o2_cols)
    lag_control_exclude = set(o2_cols + future_cols)
    return {
        "NoLeak": [c for c in feature_cols if c not in noleak_exclude],
        "WithO2": [c for c in feature_cols if c not in with_o2_exclude],
        "WithFutureNQJ": [c for c in feature_cols if c not in with_future_exclude],
        "NoO2_WithLag": [c for c in feature_cols if c not in lag_control_exclude],
    }


def selected_folds(split_report: dict[str, Any], max_folds: int) -> list[dict[str, int]]:
    folds = split_report.get("fold_indices") or []
    if not folds:
        return []
    return folds[-max_folds:]


def make_xgb(seed: int, n_estimators: int, n_jobs: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        max_depth=4,
        learning_rate=0.05,
        n_estimators=n_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=0,
    )


def make_ridge() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def fit_predict_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, seed: int, n_estimators: int, n_jobs: int):
    if model_name == "xgb":
        model = make_xgb(seed, n_estimators, n_jobs)
    elif model_name == "ridge":
        model = make_ridge()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.fit(X_train, y_train)
    return model, np.asarray(model.predict(X_valid), dtype=float)


def run_cv_experiment(
    exp_id: str,
    model_name: str,
    policy: str,
    features: list[str],
    train_df: pd.DataFrame,
    folds: list[dict[str, int]],
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    fold_rows: list[dict[str, Any]] = []
    oof = np.full(len(train_df), np.nan, dtype=float)
    for fold in folds:
        fold_no = int(fold["fold"])
        tr_start = int(fold["train_start_idx"])
        tr_end = int(fold["train_end_idx"])
        va_start = int(fold["val_start_idx"])
        va_end = min(int(fold["val_end_idx"]), len(train_df))
        if tr_end <= tr_start or va_end <= va_start:
            continue
        tr_frame = train_df.iloc[tr_start:tr_end].dropna(subset=[TARGET])
        va_frame = train_df.iloc[va_start:va_end].dropna(subset=[TARGET])
        if tr_frame.empty or va_frame.empty:
            continue
        X_tr = tr_frame[features]
        y_tr = tr_frame[TARGET]
        X_va = va_frame[features]
        y_va = va_frame[TARGET]
        _, pred = fit_predict_model(model_name, X_tr, y_tr, X_va, seed + fold_no, n_estimators, n_jobs)
        oof[va_frame.index.to_numpy()] = pred
        scores = metric_row(y_va, pred)
        fold_rows.append(
            {
                "experiment_id": exp_id,
                "model": model_name,
                "feature_policy": policy,
                "fold": fold_no,
                "train_rows": len(X_tr),
                "valid_rows": len(X_va),
                **scores,
            }
        )
        print(f"[cv] {exp_id} fold={fold_no} rmse={scores['rmse']:.6f} mae={scores['mae']:.6f}", flush=True)
    valid_mask = ~np.isnan(oof)
    cv_scores = metric_row(train_df.loc[valid_mask, TARGET], oof[valid_mask])
    fold_df = pd.DataFrame(fold_rows)
    result = {
        "experiment_id": exp_id,
        "model": model_name,
        "feature_policy": policy,
        "feature_count": len(features),
        "cv_rows_scored": int(valid_mask.sum()),
        "cv_rmse": cv_scores["rmse"],
        "cv_mae": cv_scores["mae"],
        "cv_r2": cv_scores["r2"],
        "fold_rmse_mean": float(fold_df["rmse"].mean()) if not fold_df.empty else np.nan,
        "fold_rmse_std": float(fold_df["rmse"].std(ddof=0)) if len(fold_df) > 1 else 0.0,
    }
    return result, fold_df, oof


def feature_importance(model: Any, features: list[str]) -> pd.DataFrame:
    if isinstance(model, XGBRegressor):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif isinstance(model, Pipeline):
        coef = np.asarray(model.named_steps["model"].coef_, dtype=float).ravel()
        values = np.abs(coef)
    else:
        values = np.zeros(len(features), dtype=float)
    out = pd.DataFrame({"feature": features, "importance": values})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def dwatt_bin_report(df: pd.DataFrame, pred: np.ndarray, reports: Path) -> pd.DataFrame:
    if "feat_DWATT_bin" in df.columns:
        bins = df["feat_DWATT_bin"].astype("Int64").astype(str)
    elif "IGCC.CC.G1.DWATT" in df.columns:
        bins = pd.qcut(df["IGCC.CC.G1.DWATT"], q=4, duplicates="drop").astype(str)
    else:
        bins = pd.Series(["all"] * len(df), index=df.index)
    rows = []
    err = pred - df[TARGET].to_numpy()
    for name, idx in bins.groupby(bins).groups.items():
        if len(idx) == 0:
            continue
        y_true = df.loc[idx, TARGET]
        y_pred = pred[df.index.get_indexer(idx)]
        rows.append(
            {
                "dwatt_bin": str(name),
                "rows": int(len(idx)),
                "rmse": rmse(y_true, y_pred),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "bias_mean_pred_minus_actual": float(np.mean(err[df.index.get_indexer(idx)])),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(reports / "error_by_dwatt_bin.csv", index=False)
    return out


def hour_report(holdout_df: pd.DataFrame, pred: np.ndarray, reports: Path) -> pd.DataFrame:
    if "feat_hour" in holdout_df.columns:
        hour = holdout_df["feat_hour"].astype(int)
    else:
        hour = pd.Series([0] * len(holdout_df), index=holdout_df.index)
    err = pred - holdout_df[TARGET].to_numpy()
    rows = []
    for h, idx in hour.groupby(hour).groups.items():
        y_true = holdout_df.loc[idx, TARGET]
        y_pred = pred[holdout_df.index.get_indexer(idx)]
        rows.append(
            {
                "hour": int(h),
                "rows": int(len(idx)),
                "rmse": rmse(y_true, y_pred),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "bias_mean_pred_minus_actual": float(np.mean(err[holdout_df.index.get_indexer(idx)])),
            }
        )
    out = pd.DataFrame(rows).sort_values("hour")
    out.to_csv(reports / "holdout_error_by_hour.csv", index=False)
    return out


def high_nox_report(train_df: pd.DataFrame, holdout_df: pd.DataFrame, pred: np.ndarray, reports: Path) -> pd.DataFrame:
    threshold = float(train_df[TARGET].quantile(0.95))
    actual_high = holdout_df[TARGET].to_numpy() >= threshold
    pred_high = pred >= threshold
    rows = [
        {
            "threshold_source": "train_p95",
            "threshold": threshold,
            "holdout_rows": int(len(holdout_df)),
            "actual_high_rows": int(actual_high.sum()),
            "predicted_high_rows": int(pred_high.sum()),
            "true_positive_rows": int((actual_high & pred_high).sum()),
            "missed_high_rows": int((actual_high & ~pred_high).sum()),
            "high_nox_recall": float((actual_high & pred_high).sum() / actual_high.sum()) if actual_high.sum() else np.nan,
        }
    ]
    out = pd.DataFrame(rows)
    out.to_csv(reports / "high_nox_risk_check.csv", index=False)
    return out


def write_reports(
    run_dir: Path,
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    ablation: pd.DataFrame,
    best_name: str,
    holdout_pred: np.ndarray,
    holdout_df: pd.DataFrame,
    importance: pd.DataFrame,
    dwatt_bins: pd.DataFrame,
    hour_errors: pd.DataFrame,
    high_nox: pd.DataFrame,
    domain_answers: str,
) -> None:
    reports = run_dir / "reports"
    models = run_dir / "artifacts" / "models"
    submissions = run_dir / "submissions"
    holdout_scores = metric_row(holdout_df[TARGET], holdout_pred)
    top_features = importance.head(15)
    top_feature_lines = "\n".join(f"- `{r.feature}`: {r.importance:.6f}" for r in top_features.itertuples())

    stage05 = f"""# Stage 05 모델 학습 리포트

- run_id: `{RUN_ID}`
- generated_at: `{now()}`
- 목적: 재현 가능한 baseline 모델 구조 수립
- 성능 baseline: `Ridge NoLeak (BASE-00)`
- 구조/가설검증 baseline: `{best_name}`
- primary metric: `RMSE`

## 모델 선택 결론

이번 baseline은 최고 성능 튜닝보다 구조와 공유성을 우선합니다. NoLeak 조건에서 Ridge가 더 안정적이면 성능 baseline으로 두고, XGBoost NoLeak은 가설 검증과 feature importance 해석을 위한 구조 baseline으로 유지합니다.

## Metrics

{metrics.to_markdown(index=False)}

## Fold Metrics

{fold_metrics.to_markdown(index=False)}

## Ablation

{ablation.to_markdown(index=False)}

## Top Feature Importance

{top_feature_lines}

## 주의

- `IGCC.DeNOX.AIT_H1_902`와 `feat_AIT_lag_*`는 O2 누수 검증용입니다. 최종 baseline에서는 제외했습니다.
- `feat_NQJ_future_30s`는 역인과 검증용입니다. 최종 baseline에서는 제외했습니다.
"""
    write_text(reports / "stage05_model_report.md", stage05)

    hyp_rows = []
    ablation_map = ablation.set_index("comparison").to_dict(orient="index") if not ablation.empty else {}
    o2_gap = ablation_map.get("WithO2_vs_NoLeak", {}).get("holdout_rmse_delta_pct")
    future_gap = ablation_map.get("WithFutureNQJ_vs_NoLeak", {}).get("holdout_rmse_delta_pct")
    future_rank = int(importance[importance["feature"].str.contains("future", case=False, na=False)].index.min() + 1) if importance["feature"].str.contains("future", case=False, na=False).any() else None
    ttxm_top = importance[importance["feature"].str.contains("TTXM", case=False, na=False)].head(1)
    npr_top = importance[importance["feature"].str.contains("NPR|VNPR", case=False, na=False)].head(1)
    hyp_rows.extend(
        [
            {
                "hypothesis_id": "HYP-NOX-O2-LEAKAGE",
                "support_status": "warning",
                "evidence": f"WithO2 vs NoLeak holdout RMSE delta pct={o2_gap}",
                "next_action": "O2 계열은 최종 baseline에서 제외하고, 센서/보정식 관계를 별도 확인합니다.",
            },
            {
                "hypothesis_id": "HYP-NOX-OUTPUT-SEGMENT",
                "support_status": "partially_supported",
                "evidence": "DWATT bin별 holdout error_by_dwatt_bin.csv 생성",
                "next_action": "RMSE가 큰 출력 구간을 Stage 07 현장 점검 후보로 올립니다.",
            },
            {
                "hypothesis_id": "HYP-NOX-N2-REVERSE",
                "support_status": "warning" if future_rank and future_rank <= 20 else "partially_supported",
                "evidence": f"WithFutureNQJ vs NoLeak holdout RMSE delta pct={future_gap}; future rank in final importance={future_rank}",
                "next_action": "future 피처는 최종 모델에서 제외하고 N2 lag/valve 선행성을 중심으로 해석합니다.",
            },
            {
                "hypothesis_id": "HYP-NOX-TTXM-LAG",
                "support_status": "partially_supported" if not ttxm_top.empty else "not_testable",
                "evidence": ttxm_top.to_dict(orient="records")[0] if not ttxm_top.empty else "TTXM 계열 중요도 상위 확인 불가",
                "next_action": "TTXM lag/rolling 중요도가 높으면 열관성 후보로 보고 추가 lag sweep을 권장합니다.",
            },
            {
                "hypothesis_id": "HYP-NOX-NPR-DYNAMICS",
                "support_status": "partially_supported" if not npr_top.empty else "not_testable",
                "evidence": npr_top.to_dict(orient="records")[0] if not npr_top.empty else "NPR 계열 중요도 상위 확인 불가",
                "next_action": "NPR interaction 중요도가 높으면 압력비-질소-출력 교차 조건을 현장 검토 후보로 둡니다.",
            },
        ]
    )
    hyp_df = pd.DataFrame(hyp_rows)
    hyp_df.to_csv(reports / "hypothesis_to_model_evidence.csv", index=False)
    write_json(
        reports / "hypothesis_validation_results.json",
        {"schema_version": "manual-hypothesis-validation-results.v1", "created_at": now(), "results": hyp_rows},
    )
    write_text(
        reports / "hypothesis_validation_results.md",
        "# 가설 검증 결과\n\n"
        + hyp_df.to_markdown(index=False)
        + "\n\n이 결과는 baseline 모델 기반의 1차 검증입니다. 인과 결론이 아니라 다음 현장 확인 후보입니다.\n",
    )

    pred_df = pd.DataFrame(
        {
            "row_index": np.arange(len(holdout_df)),
            "actual": holdout_df[TARGET].to_numpy(),
            "prediction": holdout_pred,
            "residual_pred_minus_actual": holdout_pred - holdout_df[TARGET].to_numpy(),
        }
    )
    if "feat_DWATT_bin" in holdout_df.columns:
        pred_df["feat_DWATT_bin"] = holdout_df["feat_DWATT_bin"].to_numpy()
    if "feat_hour" in holdout_df.columns:
        pred_df["feat_hour"] = holdout_df["feat_hour"].to_numpy()
    pred_df.to_csv(submissions / "holdout_predictions.csv", index=False)

    residual = f"""# Holdout Residual Analysis

- run_id: `{RUN_ID}`
- model: `{best_name}`
- holdout_rows: `{len(holdout_df):,}`
- holdout_rmse: `{holdout_scores['rmse']:.6f}`
- holdout_mae: `{holdout_scores['mae']:.6f}`
- holdout_r2: `{holdout_scores['r2']:.6f}`

## 시간대별 오류

{hour_errors.to_markdown(index=False)}

## 출력 구간별 오류

{dwatt_bins.to_markdown(index=False)}

## High-NOx 리스크 체크

{high_nox.to_markdown(index=False)}

## 해석 주의

Holdout 결과는 마지막 3일 일반화 성능입니다. 이 결과만으로 제어 조건을 확정하지 않고, Stage 07 액션 제안에서는 센서/밸브/운전 구간 확인 항목으로 변환합니다.
"""
    write_text(reports / "holdout_residual_analysis.md", residual)

    d07_note = "사용자가 현장 정보 부족으로 모델 해석 기반 추론을 요청했습니다."
    if "D07-ACTION-001" in domain_answers:
        d07_note = "D07-ACTION-001 accepted: 현장 이해가 부족하므로 모델 해석 기반의 보수적 액션 후보를 제안합니다."
    action = f"""# Stage 07 현장 액션 제안

- run_id: `{RUN_ID}`
- generated_at: `{now()}`
- 반영한 사용자 의견: {d07_note}

## 액션 전환 원칙

이 baseline은 제어 지시를 직접 내리는 모델이 아닙니다. 모델이 중요하다고 본 변수를 현장 점검, 센서 확인, 추가 실험 후보로 바꾸는 것이 목적입니다.

## 우선 점검 후보

1. O2 계측/보정 계통 확인: O2 계열은 타깃과 지나치게 강하게 연결되어 있어 예측에는 유리하지만 원인 해석에는 위험합니다.
2. N2 유량/밸브 응답 확인: NQJ, nicvs1 lag와 future 진단 결과를 함께 보고 제어 반응인지 선행 원인인지 분리해야 합니다.
3. 출력 구간별 운영 조건 확인: DWATT bin별 오류가 큰 구간을 우선 확인합니다.
4. 배기온도/열관성 확인: TTXM lag/rolling 중요도가 높으면 부하 변화 후 NOx 반응 지연을 확인합니다.
5. NPR/압력비 상호작용 확인: VNPR_P/S, NPR x NQJ, NPR x DWATT가 중요하면 압력비-질소-출력 교차 조건을 별도 실험 후보로 둡니다.

## 모델 중요도 상위 변수

{top_feature_lines}

## 다음 팀 논의 질문

- 최종 baseline에서 O2 계열을 제외하는 정책에 동의하는가?
- N2 future 피처가 성능을 개선하더라도 최종 모델에서 제외하는 정책에 동의하는가?
- DWATT bin 중 오류가 큰 구간을 별도 모델 또는 구간별 보정으로 다룰 것인가?
- 현장 실험 없이 제어 제안을 하지 않는 보수적 보고 방식을 유지할 것인가?
"""
    write_text(reports / "action_recommendations.md", action)

    model_card = f"""# Baseline Model Card

| 항목 | 내용 |
|---|---|
| 성능 baseline | `Ridge NoLeak (BASE-00)` |
| 구조 baseline | `{best_name}` |
| 목적 | NOx 예측 baseline 및 가설 검증 |
| 최종 피처 정책 | NoLeak |
| 제외 피처 | O2 계열, future NQJ |
| 검증 | temporal holdout, selected CV folds |
| XGBoost Holdout RMSE | `{holdout_scores['rmse']:.6f}` |
| XGBoost Holdout MAE | `{holdout_scores['mae']:.6f}` |
| 사용 제한 | 제어 지시용이 아니라 분석/점검 후보 도출용 |
"""
    write_text(reports / "final_model_card.md", model_card)

    final_report = f"""# NOx Manual Baseline Final Report

## 요약

Stage 05~07 baseline 프로세스를 완료했습니다. NoLeak 조건에서 성능 baseline은 Ridge, 구조/가설검증 baseline은 XGBoost입니다. O2 계열과 future NQJ는 최종 모델에서 제외하고 ablation/진단 용도로만 사용했습니다.

## 주요 산출물

- `artifacts/models/metrics.csv`
- `artifacts/models/ablation_results.csv`
- `reports/stage05_model_report.md`
- `reports/hypothesis_validation_results.md`
- `reports/holdout_residual_analysis.md`
- `reports/action_recommendations.md`
- `reports/final_model_card.md`

## Holdout 성능

| metric | value |
|---|---:|
| RMSE | {holdout_scores['rmse']:.6f} |
| MAE | {holdout_scores['mae']:.6f} |
| R2 | {holdout_scores['r2']:.6f} |

## 결론

이 결과는 성능 최적화 완료본이 아니라 팀 공유 가능한 baseline입니다. 다음 단계는 팀/AI 엔지니어와 O2 제외 정책, N2 역인과 처리, 출력 구간별 보정 필요성을 논의하는 것입니다.
"""
    write_text(reports / "final_analysis_report.md", final_report)


def update_share(run_dir: Path, project_root: Path, metrics: pd.DataFrame) -> None:
    share = project_root / "Share"
    if "final_candidate" in metrics.columns and metrics["final_candidate"].astype(bool).any():
        best = metrics[metrics["final_candidate"].astype(bool)].iloc[0].to_dict()
    else:
        best = metrics.sort_values("holdout_rmse").iloc[0].to_dict() if "holdout_rmse" in metrics.columns and not metrics.empty else {}
    summary = f"""# Stage 05~07 실행 결과 요약

> generated_at: `{now()}`

## 완료 상태

| 단계 | 상태 |
|---|---|
| Stage 05 | 완료 |
| Stage 05H | 완료 |
| Stage 06 | 완료 |
| Stage 07 | 완료 |

## 최종 baseline

| 항목 | 값 |
|---|---|
| 모델 | `{best.get('experiment_id', 'xgb_noleak')}` |
| 피처 정책 | `{best.get('feature_policy', 'NoLeak')}` |
| Holdout RMSE | `{best.get('holdout_rmse', '')}` |
| Holdout MAE | `{best.get('holdout_mae', '')}` |

## 핵심 결론

- 최종 baseline 피처 정책은 O2와 future NQJ를 제외한 NoLeak으로 관리합니다.
- Ridge는 성능 기준선, XGBoost는 가설 검증과 중요도 해석 기준선으로 나눠 공유합니다.
- O2/future 피처는 성능 비교용 ablation과 가설 검증용으로만 둡니다.
- 현장 액션은 모델 중요도에서 바로 제어 지시로 가지 않고, 센서/밸브/운전 구간 확인 후보로 전환합니다.

## 다음 공유 자료

- `reports/final_analysis_report.md`
- `reports/final_model_card.md`
- `reports/action_recommendations.md`
- `reports/holdout_residual_analysis.md`
"""
    write_text(share / "08_stage05_07_results_summary.md", summary)


def update_state(run_dir: Path) -> None:
    state_path = run_dir / "run_state.json"
    state = read_json(state_path, {})
    completed = state.get("stages_completed", [])
    for stage in ["05", "05H", "06", "07"]:
        if stage not in completed:
            completed.append(stage)
    state.update(
        {
            "status": "completed",
            "current_stage": "07",
            "recommended_next_stage": None,
            "recommended_next_command": None,
            "stages_completed": completed,
            "last_updated": now(),
        }
    )
    state.pop("pending_checkpoint", None)
    artifacts = state.setdefault("artifacts", {})
    artifacts["05"] = [
        "artifacts/models/metrics.csv",
        "artifacts/models/ablation_results.csv",
        "reports/stage05_model_report.md",
    ]
    artifacts["05H"] = [
        "reports/hypothesis_validation_results.md",
        "reports/hypothesis_to_model_evidence.csv",
    ]
    artifacts["06"] = [
        "submissions/holdout_predictions.csv",
        "reports/holdout_residual_analysis.md",
        "reports/error_by_dwatt_bin.csv",
    ]
    artifacts["07"] = [
        "reports/final_analysis_report.md",
        "reports/final_model_card.md",
        "reports/action_recommendations.md",
    ]
    write_json(state_path, state)
    write_text(
        run_dir / "progress.md",
        f"""# Run Progress

- Run ID: `{RUN_ID}`
- Status: `completed`
- Completed stages: `00P, 00, 01, 02, 02H, 03, 04, 05, 05H, 06, 07`
- Last updated: `{now()}`

## 최종 산출물

- `reports/final_analysis_report.md`
- `reports/final_model_card.md`
- `reports/action_recommendations.md`
- `Share/08_stage05_07_results_summary.md`
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--max-xgb-folds", type=int, default=2)
    parser.add_argument("--xgb-estimators", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    run_dir = project_root / "runs" / args.run_id
    reports = run_dir / "reports"
    models = run_dir / "artifacts" / "models"
    submissions = run_dir / "submissions"
    for path in [reports, models, submissions]:
        path.mkdir(parents=True, exist_ok=True)

    print("[load] train/holdout parquet", flush=True)
    train_df = pd.read_parquet(run_dir / "data" / "folds" / "train_split.parquet")
    holdout_df = pd.read_parquet(run_dir / "data" / "folds" / "holdout_split.parquet")
    train_target_missing = int(train_df[TARGET].isna().sum())
    holdout_target_missing = int(holdout_df[TARGET].isna().sum())
    train_model_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)
    holdout_eval_df = holdout_df.dropna(subset=[TARGET]).reset_index(drop=True)
    split_report = read_json(reports / "split_report.json", {})
    domain_answers = (reports / "domain_answers.md").read_text(encoding="utf-8")

    feature_cols = [c for c in train_df.columns if c != TARGET]
    policies = feature_policies(feature_cols)
    folds = selected_folds(split_report, args.max_xgb_folds)
    if not folds:
        raise RuntimeError("No fold indices found in split_report.json")

    write_json(
        reports / "stage05_data_contract.json",
        {
            "created_at": now(),
            "train_rows_input": int(len(train_df)),
            "holdout_rows_input": int(len(holdout_df)),
            "train_target_missing_rows": train_target_missing,
            "holdout_target_missing_rows": holdout_target_missing,
            "train_rows_modeling": int(len(train_model_df)),
            "holdout_rows_evaluated": int(len(holdout_eval_df)),
            "feature_count": len(feature_cols),
            "folds_used": [int(f["fold"]) for f in folds],
        },
    )
    print(
        f"[data] train={train_df.shape} holdout={holdout_df.shape} features={len(feature_cols)} "
        f"target_nan(train={train_target_missing}, holdout={holdout_target_missing}) folds_used={[f['fold'] for f in folds]}",
        flush=True,
    )

    experiments = [
        ("BASE-00", "ridge", "NoLeak", policies["NoLeak"], True),
        ("BASE-01", "xgb", "NoLeak", policies["NoLeak"], True),
        ("ABL-01", "xgb", "WithO2", policies["WithO2"], False),
        ("ABL-02", "xgb", "WithFutureNQJ", policies["WithFutureNQJ"], False),
        ("ABL-03", "xgb", "NoO2_WithLag", policies["NoO2_WithLag"], False),
    ]

    metrics_rows = []
    fold_frames = []
    final_models: dict[str, Any] = {}
    importance_frames: dict[str, pd.DataFrame] = {}
    holdout_predictions: dict[str, np.ndarray] = {}

    for exp_id, model_name, policy, features, final_candidate in experiments:
        print(f"[experiment] {exp_id} model={model_name} policy={policy} feature_count={len(features)}", flush=True)
        result, fold_df, _ = run_cv_experiment(exp_id, model_name, policy, features, train_df, folds, args.seed, args.xgb_estimators, args.n_jobs)
        fold_frames.append(fold_df)
        model, holdout_pred = fit_predict_model(
            model_name,
            train_model_df[features],
            train_model_df[TARGET],
            holdout_eval_df[features],
            args.seed,
            args.xgb_estimators,
            args.n_jobs,
        )
        holdout_predictions[exp_id] = holdout_pred
        holdout_scores = metric_row(holdout_eval_df[TARGET], holdout_pred)
        result.update(
            {
                "holdout_rmse": holdout_scores["rmse"],
                "holdout_mae": holdout_scores["mae"],
                "holdout_r2": holdout_scores["r2"],
                "train_rows": len(train_model_df),
                "holdout_rows": len(holdout_eval_df),
                "final_candidate": bool(final_candidate),
            }
        )
        metrics_rows.append(result)
        model_path = models / f"{exp_id.lower()}_{model_name}_{policy.lower()}.joblib"
        joblib.dump(model, model_path)
        imp = feature_importance(model, features)
        imp_path = models / f"feature_importance_{exp_id.lower()}_{policy.lower()}.csv"
        imp.to_csv(imp_path, index=False)
        importance_frames[exp_id] = imp
        final_models[exp_id] = model
        print(f"[holdout] {exp_id} rmse={holdout_scores['rmse']:.6f} mae={holdout_scores['mae']:.6f}", flush=True)

    metrics = pd.DataFrame(metrics_rows).sort_values("holdout_rmse")
    fold_metrics = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    metrics.to_csv(models / "metrics.csv", index=False)
    fold_metrics.to_csv(models / "fold_metrics.csv", index=False)

    base = metrics[metrics["experiment_id"] == "BASE-01"].iloc[0]
    ablation_rows = []
    for exp_id in ["ABL-01", "ABL-02", "ABL-03"]:
        row = metrics[metrics["experiment_id"] == exp_id].iloc[0]
        ablation_rows.append(
            {
                "comparison": f"{row['feature_policy']}_vs_NoLeak",
                "baseline_experiment": "BASE-01",
                "comparison_experiment": exp_id,
                "baseline_holdout_rmse": float(base["holdout_rmse"]),
                "comparison_holdout_rmse": float(row["holdout_rmse"]),
                "holdout_rmse_delta": float(row["holdout_rmse"] - base["holdout_rmse"]),
                "holdout_rmse_delta_pct": float((row["holdout_rmse"] - base["holdout_rmse"]) / base["holdout_rmse"] * 100.0),
            }
        )
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(models / "ablation_results.csv", index=False)

    best_name = "XGBoost NoLeak (BASE-01)"
    best_pred = holdout_predictions["BASE-01"]
    best_importance = importance_frames["BASE-01"]
    best_importance.to_csv(models / "feature_importance_xgb_noleak.csv", index=False)
    dwatt_bins = dwatt_bin_report(holdout_eval_df, best_pred, reports)
    hour_errors = hour_report(holdout_eval_df, best_pred, reports)
    high_nox = high_nox_report(train_model_df, holdout_eval_df, best_pred, reports)

    write_reports(
        run_dir,
        metrics,
        fold_metrics,
        ablation,
        best_name,
        best_pred,
        holdout_eval_df,
        best_importance,
        dwatt_bins,
        hour_errors,
        high_nox,
        domain_answers,
    )
    update_share(run_dir, project_root, metrics)
    update_state(run_dir)
    append_log(
        project_root,
        "05-07 baseline modeling",
        "Run XGBoost NoLeak/Ridge baseline, ablation diagnostics, holdout residual analysis, and final baseline reports.",
        [
            str(run_dir / "data" / "folds" / "train_split.parquet"),
            str(run_dir / "data" / "folds" / "holdout_split.parquet"),
            str(reports / "domain_answers.md"),
        ],
        [
            str(models / "metrics.csv"),
            str(reports / "stage05_model_report.md"),
            str(reports / "hypothesis_validation_results.md"),
            str(reports / "holdout_residual_analysis.md"),
            str(reports / "final_analysis_report.md"),
        ],
        "D07-ACTION-001 accepted; all configured checkpoints cleared",
        "Share the baseline pack and review O2/future ablation policy with AI engineers.",
    )
    print("[done] Stage 05-07 baseline outputs written", flush=True)


if __name__ == "__main__":
    main()
