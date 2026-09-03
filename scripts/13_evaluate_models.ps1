# Project AEGIS — [13] Multi-Model Evaluation Suite (PowerShell)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

Write-Host "=== PROJECT AEGIS: RUNNING FULL AUDIO METRICS EVALUATION SUITE ===" -ForegroundColor Cyan
python -m training.scripts.evaluate_models $args
Write-Host "=== EVALUATION COMPLETED ===" -ForegroundColor Green
