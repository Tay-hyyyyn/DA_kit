param(
    [Parameter(Mandatory=$true)][string]$ProjectRoot,
    [Parameter(Mandatory=$true)][string]$RunId,
    [Parameter(Mandatory=$true)][string]$RawDataDir,
    [Parameter(Mandatory=$true)][string]$File1,
    [Parameter(Mandatory=$true)][string]$File2,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$scripts = Join-Path $ProjectRoot "scripts"

& $PythonExe (Join-Path $scripts "stage00p_raw_intake.py") `
    --project-root $ProjectRoot `
    --run-id $RunId `
    --served-data-dir $RawDataDir `
    --file1 $File1 `
    --file2 $File2

& $PythonExe (Join-Path $scripts "stage03_feature_builder.py") `
    --project-root $ProjectRoot `
    --run-id $RunId

& $PythonExe (Join-Path $scripts "stage04_temporal_split.py") `
    --project-root $ProjectRoot `
    --run-id $RunId `
    --holdout-days 3 `
    --gap-seconds 300 `
    --cv-folds 5 `
    --start-time "2026-01-01 00:00:00"

& $PythonExe (Join-Path $scripts "stage05_to_07_baseline.py") `
    --project-root $ProjectRoot `
    --run-id $RunId `
    --max-xgb-folds 2 `
    --xgb-estimators 160 `
    --n-jobs -1

Write-Host "[done] DA_kit pipeline completed for run: $RunId"
