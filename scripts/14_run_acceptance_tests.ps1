# Project AEGIS — [14] Mission-Critical Defence Acceptance Test Runner (PowerShell)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

Write-Host "=== PROJECT AEGIS: RUNNING MISSION-CRITICAL DEFENCE ACCEPTANCE SUITE ===" -ForegroundColor Cyan
python -m pytest -v "$RootDir/tests/test_defence_mission_critical_acceptance.py" $args
Write-Host "=== DEFENCE ACCEPTANCE TESTS PASSED ===" -ForegroundColor Green
