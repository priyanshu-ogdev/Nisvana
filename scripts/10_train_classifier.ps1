# Project AEGIS — [10] Train Model 4: Acoustic Environment & Gating Classifier (PowerShell)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

Write-Host "=== PROJECT AEGIS: TRAINING MODEL 4 (aegis-clf-gate) ===" -ForegroundColor Cyan
python -m training.scripts.train_classifier $args
Write-Host "=== MODEL 4 TRAINING COMPLETED ===" -ForegroundColor Green
