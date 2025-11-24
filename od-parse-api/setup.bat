@echo off
REM Setup script for od-parse-api on Windows
REM This script installs all required dependencies

echo Installing dependencies for od-parse-api...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ and try again
    exit /b 1
)

REM Install dependencies
echo Installing from requirements.txt...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)

echo.
echo Setup completed successfully!
echo.
echo To run the API, use:
echo   python -m uvicorn app.main:app --reload
echo.
echo Or use the run.bat script:
echo   run.bat
echo.

