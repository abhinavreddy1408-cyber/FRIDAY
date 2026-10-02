# run.ps1 -- Start F.R.I.D.A.Y. using the project .venv
# Always uses the explicit venv Python so PATH order never matters.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = Get-Location }

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

# Auto-setup if venv is missing
if (-not (Test-Path $VenvPython)) {
    Write-Host "Virtual environment not found -- running setup.ps1 ..." -ForegroundColor Yellow
    & (Join-Path $ProjectRoot "setup.ps1")
}

$ollamaProcess = Get-Process ollama -ErrorAction SilentlyContinue
if (-not $ollamaProcess) {
    Write-Host "Starting Ollama..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

Write-Host "Starting F.R.I.D.A.Y. ..." -ForegroundColor Green
Write-Host "Python: $VenvPython" -ForegroundColor Gray
& $VenvPython (Join-Path $ProjectRoot "main.py")
