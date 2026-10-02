# test.ps1 -- Run pytest using the project .venv
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

Write-Host "Running tests ..." -ForegroundColor Green
Write-Host "Python: $VenvPython" -ForegroundColor Gray
& $VenvPython -m pytest
