# DA_kit

`DA_kit` is a clean, public baseline workflow for staged tabular data analysis. It packages the reusable NOx-style workflow without local machine paths, raw data, run outputs, model binaries, or private experiment history.

## What Is Included

- raw table intake and normalization
- feature-building stage
- temporal holdout and rolling fold split
- baseline model/report stage
- schema-only example data
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
