# Project AEGIS — [15] ONNX Edge Model Exporter & Latency Profiler (PowerShell)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

Write-Host "=== PROJECT AEGIS: EXPORTING MODEL TO ONNX FOR EDGE HARDWARE ===" -ForegroundColor Cyan
python -m inference.scripts.export_onnx $args
Write-Host "=== ONNX EXPORT COMPLETED ===" -ForegroundColor Green
