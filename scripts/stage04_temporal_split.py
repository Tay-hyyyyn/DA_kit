from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


TARGET = "IGCC.DeNOX.AT_H1_901_PV"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 04 temporal holdout split")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--holdout-days", type=int, default=3)
    parser.add_argument("--gap-seconds", type=int, default=300)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--start-time", default="2025-08-11 00:00:00")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    run_dir = project_root / "runs" / args.run_id
    featured_path = run_dir / "data" / "featured" / "featured_train.parquet"
    if not featured_path.exists():
        raise FileNotFoundError(f"Input not found: {featured_path}")

    folds_dir = run_dir / "data" / "folds"
    reports_dir = run_dir / "reports"
    folds_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(featured_path)
    n = len(df)
    start_time = pd.Timestamp(args.start_time)
    df["_ts_index"] = pd.date_range(start=start_time, periods=n, freq="1s")

    end_time = df["_ts_index"].max()
    holdout_start = end_time - timedelta(days=args.holdout_days)
    gap_end = holdout_start - timedelta(seconds=1)
    gap_start = holdout_start - timedelta(seconds=args.gap_seconds)

    mask_train = df["_ts_index"] < gap_start
    mask_gap = (df["_ts_index"] >= gap_start) & (df["_ts_index"] <= gap_end)
    mask_holdout = df["_ts_index"] >= holdout_start

    train_df = df.loc[mask_train].copy()
    holdout_df = df.loc[mask_holdout].copy()
    gap_count = int(mask_gap.sum())

    # Fold design aligned with the existing workflow
    train_n = len(train_df)
    fold_size = train_n // (args.cv_folds + 1)
    fold_indices = []
    for i in range(args.cv_folds):
        val_start = (i + 1) * fold_size
        val_end = min((i + 2) * fold_size, train_n)
        tr_end = max(0, val_start - args.gap_seconds)
        fold_indices.append(
            {
                "fold": i + 1,
                "train_start_idx": 0,
                "train_end_idx": int(tr_end),
                "val_start_idx": int(val_start),
                "val_end_idx": int(val_end),
                "train_rows": int(tr_end),
                "val_rows": int(max(0, val_end - val_start)),
            }
        )

    train_df = train_df.drop(columns=["_ts_index"])
    holdout_df = holdout_df.drop(columns=["_ts_index"])
    train_df.to_parquet(folds_dir / "train_split.parquet", index=False)
    holdout_df.to_parquet(folds_dir / "holdout_split.parquet", index=False)

    report = {
        "schema_version": "manual-split-report.v1",
        "created_at": now(),
        "stage": "04",
        "target_col": TARGET,
        "split_strategy": "temporal_holdout",
        "holdout_days": args.holdout_days,
        "holdout_gap_seconds": args.gap_seconds,
        "cv_folds": args.cv_folds,
        "total_rows": int(n),
        "total_features": int(df.shape[1] - 1),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "gap_rows": gap_count,
        "train_time_range": {
            "start": str(start_time),
            "end": str(start_time + timedelta(seconds=int(mask_train.sum()) - 1)),
        },
        "holdout_time_range": {
            "start": str(holdout_start),
            "end": str(end_time),
        },
        "fold_indices": fold_indices,
        "target_stats": {
            "train_mean": float(train_df[TARGET].mean()),
            "train_std": float(train_df[TARGET].std()),
            "holdout_mean": float(holdout_df[TARGET].mean()),
            "holdout_std": float(holdout_df[TARGET].std()),
            "distribution_shift_check": abs(float(train_df[TARGET].mean()) - float(holdout_df[TARGET].mean()))
            / float(train_df[TARGET].std()),
        },
    }
    (reports_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] Stage 04 train={len(train_df):,} holdout={len(holdout_df):,}", flush=True)


if __name__ == "__main__":
    main()
