from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "IGCC.DeNOX.AT_H1_901_PV"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 03 feature builder for NOx baseline")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    run_dir = project_root / "runs" / args.run_id
    processed = run_dir / "data" / "processed" / "normalized_train.parquet"
    if not processed.exists():
        raise FileNotFoundError(f"Input not found: {processed}")

    out_dir = run_dir / "data" / "featured"
    reports_dir = run_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed)
    drop_cols = ["_manual_row_id", "Column1", "IGCC.CC.G1.ttfr1"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # numeric median impute for non-target cols
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in num_cols:
        num_cols.remove(TARGET)
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Physical sanity filter
    temp_keys = ["TTXM", "CTIM", "CTD", "ATID", "NTNJ", "ndt1", "itdp", "tcsph1", "TT_H1_90123", "FTSG"]
    temp_cols = [c for c in df.columns if any(k in c for k in temp_keys)]
    for c in temp_cols:
        if c in df.columns:
            mask = pd.to_numeric(df[c], errors="coerce") < -50
            if mask.any():
                df.loc[mask, c] = df[c].median()
    if "IGCC.CC.G1.DWATT" in df.columns:
        mask = pd.to_numeric(df["IGCC.CC.G1.DWATT"], errors="coerce") < 0
        if mask.any():
            df.loc[mask, "IGCC.CC.G1.DWATT"] = df["IGCC.CC.G1.DWATT"].median()

    C = {
        "DWATT": "IGCC.CC.G1.DWATT",
        "TTXM": "IGCC.CC.G1.TTXM",
        "CTIM": "IGCC.CC.G1.CTIM",
        "CTD": "IGCC.CC.G1.CTD",
        "NQJ": "IGCC.CC.G1.NQJ",
        "nicvs1": "IGCC.CC.G1.nicvs1",
        "VNPR_P": "IGCC.CC.G1.VNPR_P",
        "VNPR_S": "IGCC.CC.G1.VNPR_S",
        "CPD": "IGCC.CC.G1.CPD",
        "csgv": "IGCC.CC.G1.csgv",
        "ca_fqsg_cl": "IGCC.CC.G1.ca_fqsg_cl",
        "AIT": "IGCC.DeNOX.AIT_H1_902",
    }

    def col(name: str) -> str:
        return C.get(name, name)

    # A. Domain-based features
    df["feat_pressure_ratio"] = df[col("VNPR_P")] / df[col("VNPR_S")].replace(0, np.nan)
    df["feat_delta_temp_exhaust_inlet"] = df[col("TTXM")] - df[col("CTIM")]
    df["feat_delta_temp_exhaust_compressor"] = df[col("TTXM")] - df[col("CTD")]
    df["feat_power_per_fuel"] = df[col("DWATT")] / df[col("ca_fqsg_cl")].replace(0, np.nan)
    df["feat_power_per_n2"] = df[col("DWATT")] / df[col("NQJ")].replace(0, np.nan)
    df["feat_npr_x_nqj"] = df[col("VNPR_P")] * df[col("NQJ")]
    df["feat_npr_x_dwatt"] = df[col("VNPR_P")] * df[col("DWATT")]

    # B. Hypothesis features
    df["feat_AIT_lag_30s"] = df[col("AIT")].shift(30)
    df["feat_AIT_lag_60s"] = df[col("AIT")].shift(60)
    dwatt = df[col("DWATT")]
    bins = [0, 155, 170, 185, float(dwatt.max()) + 1]
    df["feat_DWATT_bin"] = pd.cut(dwatt, bins=bins, labels=[0, 1, 2, 3]).astype(float)
    df["feat_hinge_DWATT_170"] = (dwatt - 170).clip(lower=0)
    df["feat_DWATT_x_NQJ"] = dwatt * df[col("NQJ")]

    for lag in [30, 60, 120, 300]:
        df[f"feat_NQJ_lag_{lag}s"] = df[col("NQJ")].shift(lag)
    df["feat_NQJ_future_30s"] = df[col("NQJ")].shift(-30)
    for lag in [30, 60, 120]:
        df[f"feat_nicvs1_lag_{lag}s"] = df[col("nicvs1")].shift(lag)

    df["feat_TTXM_lag_60s"] = df[col("TTXM")].shift(60)
    df["feat_TTXM_lag_300s"] = df[col("TTXM")].shift(300)
    df["feat_diff_TTXM"] = df[col("TTXM")].diff(1)
    df["feat_rolling_mean_TTXM_60"] = df[col("TTXM")].rolling(60, min_periods=1).mean()
    df["feat_rolling_mean_TTXM_300"] = df[col("TTXM")].rolling(300, min_periods=1).mean()
    df["feat_rolling_std_TTXM_60"] = df[col("TTXM")].rolling(60, min_periods=1).std()

    df["feat_VNPR_P_minus_S"] = df[col("VNPR_P")] - df[col("VNPR_S")]
    df["feat_VNPR_P_div_S"] = df[col("VNPR_P")] / df[col("VNPR_S")].replace(0, np.nan)
    df["feat_NPR_x_NQJ"] = df[col("VNPR_P")] * df[col("NQJ")]
    df["feat_NPR_x_DWATT"] = df[col("VNPR_P")] * df[col("DWATT")]
    npr_med = pd.to_numeric(df[col("VNPR_P")], errors="coerce").median()
    df["feat_hinge_NPR"] = (df[col("VNPR_P")] - npr_med).clip(lower=0)

    # C. Time-series
    for lag in [30, 60]:
        df[f"feat_DWATT_lag_{lag}s"] = df[col("DWATT")].shift(lag)
    df["feat_CPD_lag_30s"] = df[col("CPD")].shift(30)
    df["feat_csgv_lag_30s"] = df[col("csgv")].shift(30)
    df["feat_diff_DWATT"] = df[col("DWATT")].diff(1)
    df["feat_diff_NQJ"] = df[col("NQJ")].diff(1)
    df["feat_diff_CPD"] = df[col("CPD")].diff(1)
    df["feat_diff_nicvs1"] = df[col("nicvs1")].diff(1)
    df["feat_rolling_mean_NQJ_60"] = df[col("NQJ")].rolling(60, min_periods=1).mean()
    df["feat_rolling_mean_DWATT_300"] = df[col("DWATT")].rolling(300, min_periods=1).mean()

    # D. Time features
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["feat_hour"] = df["timestamp"].dt.hour
        df["feat_hour_sin"] = np.sin(2 * np.pi * df["feat_hour"] / 24)
        df["feat_hour_cos"] = np.cos(2 * np.pi * df["feat_hour"] / 24)
        df["feat_day_of_week"] = df["timestamp"].dt.dayofweek
        df["feat_is_night"] = ((df["feat_hour"] < 6) | (df["feat_hour"] >= 22)).astype(int)

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    for c in feat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    all_cols = [c for c in df.columns if c != TARGET]
    original_cols = [c for c in all_cols if not c.startswith("feat_")]
    derived_cols = [c for c in all_cols if c.startswith("feat_")]

    out_path = out_dir / "featured_train.parquet"
    df.to_parquet(out_path, index=False)

    manifest = {
        "schema_version": "manual-feature-manifest.v1",
        "created_at": now(),
        "stage": "03",
        "target_col": TARGET,
        "total_features": len(all_cols),
        "original_features": len(original_cols),
        "derived_features": len(derived_cols),
        "total_rows": len(df),
        "original_feature_list": original_cols,
        "derived_feature_list": derived_cols,
        "leakage_watch": ["feat_NQJ_future_30s", "feat_AIT_lag_30s", "feat_AIT_lag_60s"],
    }
    (reports_dir / "feature_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] Stage 03 features={len(all_cols)} rows={len(df):,}", flush=True)


if __name__ == "__main__":
    main()
