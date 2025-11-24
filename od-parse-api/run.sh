#!/bin/bash
# Run script for od-parse-api on Linux/Mac
# This script starts the FastAPI server using uvicorn

set -e

echo "Starting Intelligent File Parser API..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.11+ and try again"
    exit 1
fi

# Start the API server
echo "Starting uvicorn server..."
echo "API will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo
echo "Press Ctrl+C to stop the server"
echo

python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

