@echo off
REM Run script for od-parse-api on Windows
REM This script starts the FastAPI server using uvicorn

echo Starting Intelligent File Parser API...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ and try again
    exit /b 1
)

REM Start the API server
echo Starting uvicorn server...
echo API will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

