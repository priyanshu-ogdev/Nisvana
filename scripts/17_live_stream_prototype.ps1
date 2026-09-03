# Project AEGIS — [17] Real-Time Live Microphone & Headset ANC Prototype (PowerShell)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

Write-Host "=== PROJECT AEGIS: RUNNING LIVE STREAMING ANC PROTOTYPE DEMONSTRATION ===" -ForegroundColor Cyan
python -m inference.scripts.live_mic_anc $args
Write-Host "=== PROTOTYPE DEMONSTRATION FINISHED ===" -ForegroundColor Green
