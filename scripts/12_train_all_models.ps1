# Project AEGIS — [12] Master Pipeline: Train All Models Sequentially (PowerShell)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

Write-Host "========================================================================" -ForegroundColor Yellow
Write-Host "PROJECT AEGIS — SEQUENTIAL MULTI-MODEL TRAINING PIPELINE" -ForegroundColor Yellow
Write-Host "========================================================================" -ForegroundColor Yellow

Write-Host "[1/4] Training Model 1 (aegis-se-primary)..." -ForegroundColor Cyan
& "$ScriptDir\07_train_se_primary.ps1" @args

Write-Host "[2/4] Training Model 2 (aegis-se-escalation)..." -ForegroundColor Cyan
& "$ScriptDir\08_train_se_escalation.ps1" @args

Write-Host "[3/4] Training Model 3 (aegis-se-crosscheck)..." -ForegroundColor Cyan
& "$ScriptDir\09_train_se_crosscheck.ps1" @args

Write-Host "[4/4] Training Model 4 (aegis-clf-gate)..." -ForegroundColor Cyan
& "$ScriptDir\10_train_classifier.ps1" @args

Write-Host "========================================================================" -ForegroundColor Green
Write-Host "ALL AEGIS MODELS SUCCESSFULLY TRAINED AND VERIFIED" -ForegroundColor Green
Write-Host "Checkpoints banked in: $RootDir\data\checkpoints\" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Green
