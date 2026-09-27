# run_tests.ps1
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " STARTING OFFSHORE TELEMETRY PIPELINE TESTING" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Define Paths
$VENV_DIR = ".venv"
$REQ_FILE = "requirements.txt"
$SCRIPT_FILE = "src/anomaly_detector.py"

# 2. Setup Virtual Environment if it doesn't exist
if (-not (Test-Path -Path $VENV_DIR)) {
    Write-Host "[INIT] Python Virtual Environment not found. Creating one now..." -ForegroundColor Yellow
    python -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment. Ensure Python is installed and in your PATH."
        exit $LASTEXITCODE
    }
} else {
    Write-Host "[INIT] Existing Virtual Environment detected." -ForegroundColor Green
}

# 3. Activate Virtual Environment and Install Dependencies
Write-Host "[SETUP] Activating environment and checking dependencies..." -ForegroundColor Yellow
& "$VENV_DIR/Scripts/pip" install -r $REQ_FILE --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install required dependencies from $REQ_FILE."
    exit $LASTEXITCODE
}
Write-Host "[SETUP] Dependencies are up to date." -ForegroundColor Green

# 4. Execute the Anomaly Tracking Pipeline
Write-Host "[EXECUTE] Running anomaly tracking template script..." -ForegroundColor Yellow
Write-Host "--------------------------------------------------" -ForegroundColor DarkGray

# Run python script and capture output stream
& "$VENV_DIR/Scripts/python" $SCRIPT_FILE

# 5. Check Execution Status
Write-Host "--------------------------------------------------" -ForegroundColor DarkGray
if ($LASTEXITCODE -eq 0) {
    Write-Host " PIPELINE TESTING COMPLETED SUCCESSFULLY!" -ForegroundColor Green
} else {
    Write-Host " PIPELINE TESTING FAILED WITH EXIT CODE $LASTEXITCODE" -ForegroundColor Red
}
Write-Host "==================================================" -ForegroundColor Cyan
