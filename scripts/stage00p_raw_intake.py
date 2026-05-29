from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 00P raw intake for NOx baseline")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--served-data-dir", required=True)
    parser.add_argument("--file1", required=True)
    parser.add_argument("--file2", required=True)
    parser.add_argument("--meta-rows", type=int, default=5)
    parser.add_argument("--timestamp-col", default="TagName")
    parser.add_argument("--timestamp-out", default="timestamp")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    run_dir = project_root / "runs" / args.run_id
    processed_dir = run_dir / "data" / "processed"
    reports_dir = run_dir / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    path1 = Path(args.served_data_dir) / args.file1
    path2 = Path(args.served_data_dir) / args.file2
    if not path1.exists() or not path2.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path1} or {path2}")

    # Read with metadata block (first 5 rows) and data block (from row 6)
    raw1 = pd.read_csv(path1, low_memory=False)
    raw2 = pd.read_csv(path2, low_memory=False)

    if len(raw1) <= args.meta_rows or len(raw2) <= args.meta_rows:
        raise ValueError("Raw CSV row count is too small for metadata split.")

    meta1 = raw1.iloc[: args.meta_rows].copy()
    meta2 = raw2.iloc[: args.meta_rows].copy()
    data1 = raw1.iloc[args.meta_rows :].copy()
    data2 = raw2.iloc[args.meta_rows :].copy()

    # Merge strategy used in this project: vertical concat then sort by timestamp
    merged = pd.concat([data1, data2], axis=0, ignore_index=True)
    if args.timestamp_col not in merged.columns:
        raise KeyError(f"Timestamp column `{args.timestamp_col}` not found.")

    merged[args.timestamp_col] = pd.to_datetime(merged[args.timestamp_col], errors="coerce")
    merged = merged.rename(columns={args.timestamp_col: args.timestamp_out})
    merged = merged.sort_values(args.timestamp_out).reset_index(drop=True)
    merged = merged.drop_duplicates(subset=[args.timestamp_out], keep="first")

    # Convert numeric candidates
    for col in merged.columns:
        if col == args.timestamp_out:
            continue
        converted = pd.to_numeric(merged[col], errors="coerce")
        if converted.notna().mean() >= 0.6:
            merged[col] = converted

    out_csv = processed_dir / "normalized_train.csv"
    out_parquet = processed_dir / "normalized_train.parquet"
    merged.to_csv(out_csv, index=False)
    merged.to_parquet(out_parquet, index=False)

    profile = {
        "created_at": now(),
        "file1": str(path1),
        "file2": str(path2),
        "meta_rows": args.meta_rows,
        "meta_equal_shape": list(meta1.shape) == list(meta2.shape),
        "merged_rows": int(len(merged)),
        "merged_cols": int(len(merged.columns)),
        "timestamp_col": args.timestamp_out,
        "timestamp_missing_rows": int(merged[args.timestamp_out].isna().sum()),
        "timestamp_duplicate_rows_after_dedup": int(merged.duplicated(subset=[args.timestamp_out]).sum()),
        "out_csv": str(out_csv),
        "out_parquet": str(out_parquet),
    }
    write_json(reports_dir / "raw_file_profile.json", profile)

    print(f"[done] Stage 00P rows={len(merged):,} cols={len(merged.columns)}", flush=True)


if __name__ == "__main__":
    main()
