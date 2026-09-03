# Project AEGIS — [07] Train Model 1: Primary Real-Time SE (PowerShell)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

Write-Host "=== PROJECT AEGIS: TRAINING MODEL 1 (aegis-se-primary) ===" -ForegroundColor Cyan
python -m training.scripts.train_se_primary $args
Write-Host "=== MODEL 1 TRAINING COMPLETED ===" -ForegroundColor Green
