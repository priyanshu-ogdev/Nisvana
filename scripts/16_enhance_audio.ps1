# Project AEGIS — [16] Offline Audio File Enhancement Runner (PowerShell)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

Write-Host "=== PROJECT AEGIS: RUNNING AUDIO FILE ENHANCER ===" -ForegroundColor Cyan
python -m inference.scripts.enhance_audio $args
Write-Host "=== AUDIO ENHANCEMENT COMPLETED ===" -ForegroundColor Green
