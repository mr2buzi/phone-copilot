$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Starting phone-copilot controller..."
Write-Host "Open http://127.0.0.1:8765/training and use Autonomous Training Loop to start or stop the scheduler."
Write-Host "This script starts the server only. It does not create Windows Task Scheduler jobs."

python -m uvicorn apps.controller.main:app --host 127.0.0.1 --port 8765
