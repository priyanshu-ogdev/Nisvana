# Project AEGIS — [08] Train Model 2: Escalation SE (PowerShell)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

Write-Host "=== PROJECT AEGIS: TRAINING MODEL 2 (aegis-se-escalation) ===" -ForegroundColor Cyan
python -m training.scripts.train_se_escalation $args
Write-Host "=== MODEL 2 TRAINING COMPLETED ===" -ForegroundColor Green
