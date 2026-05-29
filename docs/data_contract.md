# NOx/IGCC Data Contract

## Raw Inputs

The baseline intake stage expects two NOx/IGCC CSV files with the same column layout.

- The first rows can contain metadata.
- The timestamp-like column defaults to `TagName`.
- Numeric columns are converted when most values are parseable as numbers.
- The target column defaults to `IGCC.DeNOX.AT_H1_901_PV`.
- The public sample columns are NOx/IGCC example columns, not a generic schema.

## Outputs

All generated outputs live under `runs/<run_id>/`, including processed data, feature data, fold data, reports, submissions, and model artifacts. This directory is ignored by Git.

## Public Example

`examples/schema_sample.csv` is schema-only NOx/IGCC sample data. It is not a production or private dataset.
