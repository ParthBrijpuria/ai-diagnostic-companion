@echo off
REM AI Diagnostic Companion Startup Script for Windows
REM This script starts both the Next.js frontend and Python ML service

echo 🚀 Starting AI Diagnostic Companion...

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed. Please install Node.js 18 or higher.
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Install Node.js dependencies if needed
if not exist "node_modules" (
    echo 📦 Installing Node.js dependencies...
    npm install
)

REM Start Python ML service in background
echo 🐍 Starting Python ML service...
cd ml
if not exist "venv" (
    echo 📦 Creating Python virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
pip install -r requirements.txt >nul 2>&1
start /B python diseasePredictor.py
cd ..

REM Wait a moment for ML service to start
timeout /t 3 /nobreak >nul

REM Start Next.js development server
echo ⚛️ Starting Next.js development server...
echo.
echo 🌟 Services starting:
echo    Frontend: http://localhost:3000
echo    ML Service: http://localhost:5000
echo.
echo Press Ctrl+C to stop all services

npm run dev

pause
