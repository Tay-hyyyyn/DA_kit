# GitHub Publish Guide

## Before Commit

Run:

```powershell
$privateMarkers = @(
  "C:" + "\\Users",
  "Served" + " Data",
  "2025" + "0811",
  "2025" + "0825",
  "References" + "/01",
  "." + "venv-cantera"
)
foreach ($marker in $privateMarkers) { rg -n $marker . }
python -m py_compile scripts/stage00p_raw_intake.py scripts/stage03_feature_builder.py scripts/stage04_temporal_split.py scripts/stage05_to_07_baseline.py
python -m unittest discover -s tests
git status --short
```

Expected:

- no private-path or raw-source matches,
- Python compile succeeds,
- tests pass,
- only intended public files are staged.

## Stage Explicitly

```powershell
git add README.md .gitignore requirements.txt
git add config/nox_config.example.json
git add docs/workflow_overview.md docs/github_publish_guide.md docs/data_contract.md
git add examples/schema_sample.csv
git add scripts/stage00p_raw_intake.py scripts/stage03_feature_builder.py scripts/stage04_temporal_split.py scripts/stage05_to_07_baseline.py scripts/run_share_pipeline.ps1
git add tests/test_public_package.py
```

Never use broad staging for this package.
