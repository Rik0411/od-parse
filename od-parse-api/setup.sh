#!/bin/bash
# Setup script for od-parse-api on Linux/Mac
# This script installs all required dependencies

set -e

echo "Installing dependencies for od-parse-api..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.11+ and try again"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo "Installing from requirements.txt..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo
echo "Setup completed successfully!"
echo
echo "To run the API, use:"
echo "  python3 -m uvicorn app.main:app --reload"
echo
echo "Or use the run.sh script:"
echo "  ./run.sh"
echo

