# setup.ps1 -- F.R.I.D.A.Y. environment bootstrap
# Idempotent: safe to re-run at any time.
# Uses explicit paths to the project .venv so it never accidentally
# installs into a system Python or another venv.

Set-StrictMode -Version Latest

# -- Resolve project root (directory this script lives in) ---------------------
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = Get-Location }

$VenvDir    = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip    = Join-Path $VenvDir "Scripts\pip.exe"
$ReqFile    = Join-Path $ProjectRoot "requirements.txt"

# -- 1. Execution policy ------------------------------------------------------
$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -notin @('RemoteSigned', 'Unrestricted', 'Bypass')) {
    Write-Host "[1/5] Setting execution policy to RemoteSigned for CurrentUser..." -ForegroundColor Yellow
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
} else {
    Write-Host "[1/5] Execution policy OK ($policy)" -ForegroundColor Green
}

# -- 2. Create or repair venv -------------------------------------------------
$needsCreate = $false

if (-not (Test-Path $VenvPython)) {
    $needsCreate = $true
} else {
    # Verify pip works inside the existing venv; recreate if broken
    $pipCheck = & $VenvPython -m pip --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[2/5] Existing .venv is broken (pip missing). Recreating ..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $VenvDir
        $needsCreate = $true
    } else {
        Write-Host "[2/5] Virtual environment OK at $VenvDir" -ForegroundColor Green
    }
}

if ($needsCreate) {
    Write-Host "[2/5] Creating virtual environment at $VenvDir ..." -ForegroundColor Cyan
    python -m venv $VenvDir
    if (-not (Test-Path $VenvPython)) {
        Write-Host "[ERROR] Failed to create venv. Ensure Python 3.10+ is installed and on PATH." -ForegroundColor Red
        exit 1
    }
}

$env:TRANSFORMERS_CACHE = "$PSScriptRoot\models\transformers"
New-Item -ItemType Directory -Force -Path "$PSScriptRoot\models\transformers" | Out-Null
New-Item -ItemType Directory -Force -Path "$PSScriptRoot\models\ollama" | Out-Null

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "OLLAMA NOT INSTALLED"
    Write-Host "Download from: https://ollama.ai/download"
    Write-Host "Then run: ollama pull qwen2.5:3b-instruct-q4_K_M"
    Write-Host "Then run: ollama pull qwen2.5:1.5b-instruct-q4_K_M"
    Write-Host "Then run: ollama pull qwen2.5-coder:3b-instruct-q4_K_M"
}

# -- 3. Upgrade pip inside the venv -------------------------------------------
Write-Host "[3/5] Upgrading pip inside .venv ..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip --quiet 2>&1 | Out-Host

# -- 4. Install requirements --------------------------------------------------
Write-Host "[4/5] Installing dependencies from requirements.txt ..." -ForegroundColor Cyan
& $VenvPip install -r $ReqFile 2>&1 | Out-Host

# Also install dev/test dependencies from pyproject.toml
Write-Host "      Installing dev/test dependencies ..." -ForegroundColor Cyan
& $VenvPip install pytest pytest-asyncio pytest-cov pytest-xdist 2>&1 | Out-Host

# PyAudio fallback via pipwin if the wheel did not build
$pyaudioCheck = & $VenvPython -c "import pyaudio" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "      PyAudio direct install failed -- trying pipwin fallback ..." -ForegroundColor Yellow
    & $VenvPip install pipwin 2>&1 | Out-Host
    & $VenvPython -m pipwin refresh 2>&1 | Out-Host
    & $VenvPython -m pipwin install pyaudio 2>&1 | Out-Host
    & $VenvPip install -r $ReqFile 2>&1 | Out-Host
}

# -- 5. Final verification ----------------------------------------------------
Write-Host "[5/5] Verifying critical imports ..." -ForegroundColor Cyan
$verifyResult = & $VenvPython -c "import speech_recognition; import pyaudio; print('OK')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "  Environment ready!" -ForegroundColor Green
    Write-Host "  Python : $VenvPython" -ForegroundColor Gray
    Write-Host "  To run : .\run.ps1" -ForegroundColor Gray
    Write-Host "  To test: .\test.ps1" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "[WARN] Setup finished but SpeechRecognition or PyAudio could not be imported." -ForegroundColor Yellow
    $hint = 'Try running: ' + $VenvPip + ' install SpeechRecognition PyAudio'
    Write-Host $hint -ForegroundColor Yellow
}
