# NOx/IGCC Workflow Overview

This repository is a NOx/IGCC-specific baseline workflow. It is intentionally staged so each step leaves inspectable artifacts under `runs/<run_id>/`.

| Stage | Script | Main Output |
|---|---|---|
| 00P | `scripts/stage00p_raw_intake.py` | normalized NOx/IGCC raw table |
| 03 | `scripts/stage03_feature_builder.py` | NOx/IGCC feature table and manifest |
| 04 | `scripts/stage04_temporal_split.py` | train/holdout split and fold report |
| 05-07 | `scripts/stage05_to_07_baseline.py` | NOx baseline models and reports |

The repository stores code, docs, examples, and tests only. Runtime outputs stay local.
