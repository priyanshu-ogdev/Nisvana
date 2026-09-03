# Project AEGIS — [11] Train Model 5: Gated Acoustic Echo Cancellation (PowerShell)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

Write-Host "=== PROJECT AEGIS: TRAINING MODEL 5 (aegis-aec-gate) ===" -ForegroundColor Cyan
Write-Host "Note: Default deployment uses pretrained deepvqe-ggml checkpoint."
python -m training.scripts.train_aec $args
Write-Host "=== MODEL 5 SCRIPT FINISHED ===" -ForegroundColor Green
