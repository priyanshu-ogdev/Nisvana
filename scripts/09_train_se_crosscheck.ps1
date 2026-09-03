# Project AEGIS — [09] Train Model 3: State-Space SE Cross-Check (PowerShell)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

Write-Host "=== PROJECT AEGIS: TRAINING MODEL 3 (aegis-se-crosscheck) ===" -ForegroundColor Cyan
python -m training.scripts.train_se_crosscheck $args
Write-Host "=== MODEL 3 TRAINING COMPLETED ===" -ForegroundColor Green
