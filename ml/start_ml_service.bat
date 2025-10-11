@echo off
REM Medical Report ML Service Startup Script for Windows
REM This script sets up and starts the Python Flask ML service

echo 🚀 Starting Medical Report ML Service...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Please install pip.
    pause
    exit /b 1
)

REM Navigate to ML directory
cd ml

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing Python dependencies...
pip install -r requirements.txt

REM Start the Flask service
echo 🌟 Starting Flask ML service on http://localhost:5000
echo 📊 Available endpoints:
echo    GET  /health - Health check
echo    POST /predict - Disease prediction
echo    GET  /models - List available patterns
echo    POST /analyze - Analyze metrics
echo.
echo Press Ctrl+C to stop the service

python diseasePredictor.py

pause
