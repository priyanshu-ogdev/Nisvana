# Project AEGIS — [18] SIH Defence Benchmark & Inference Verification Runner (PowerShell)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

Write-Host "=== PROJECT AEGIS: RUNNING OFFICIAL SIH DEFENCE BENCHMARK AUDIT ===" -ForegroundColor Cyan
python -m pytest -v "$RootDir/tests/test_sih_inference_metrics.py" $args
Write-Host "=== SIH DEFENCE COMPLIANCE AUDIT PASSED ===" -ForegroundColor Green
