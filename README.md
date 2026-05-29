# DA_kit NOx/IGCC Baseline

`DA_kit` is a **NOx/IGCC-specific public baseline workflow**. It is not the fully generic data-analysis kit.

This repository packages a sanitized NOx-style staged workflow without local machine paths, raw data, run outputs, model binaries, or private experiment history. The feature logic, example schema, target column, and reporting language are tailored to the NOx/IGCC analysis context.

## What Is Included

- raw NOx/IGCC table intake and normalization
- NOx/IGCC feature-building stage
- temporal holdout and rolling fold split
- NOx baseline model/report stage
- NOx/IGCC schema-only example data
- publication safety checks

## Repository Layout

```text
DA_kit/
├── config/
├── docs/
├── examples/
├── scripts/
└── tests/
```

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Prepare two raw CSV files with the same schema, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_share_pipeline.ps1 `
  -ProjectRoot . `
  -RunId demo_run `
  -RawDataDir <raw-data-dir> `
  -File1 <first-input.csv> `
  -File2 <second-input.csv> `
  -PythonExe .\.venv\Scripts\python.exe
```

Generated outputs are written under `runs/<run_id>/` and are intentionally ignored by Git.

## Safety

Do not commit raw data, run outputs, model artifacts, private configs, local virtual environments, or machine-specific paths. See `docs/github_publish_guide.md` before pushing.

## Generic Kit

For a fully generic, config-driven tabular data-analysis kit, use the separate universal Manual-based package repository rather than this NOx/IGCC baseline repository.
