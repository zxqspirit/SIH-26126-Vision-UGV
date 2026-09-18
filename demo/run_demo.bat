@echo off
REM ==============================================================================
REM SIH 26126 - TerrainSight UGV Demonstration Launcher (Windows)
REM Organization: Bharat Electronics Limited (BEL)
REM ==============================================================================

setlocal enabledelayedexpansion

echo ==================================================================
echo     TERRAINSIGHT UGV - SIH 26126 DEMONSTRATION RUNNER
echo     Bharat Electronics Limited (BEL) ^| Vision-Only Navigation
echo ==================================================================

cd /d "%~dp0\.."

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in PATH. Please install Python 3.9+.
    pause
    exit /b 1
)

echo [INFO] Starting Mission Control Dashboard Server...
echo [INFO] Open your web browser at: http://localhost:5000
echo [INFO] Press Ctrl+C to terminate.
echo ==================================================================

start "" http://localhost:5000
python src/visualization/dashboard_server.py
pause
