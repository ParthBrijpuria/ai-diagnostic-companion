#!/bin/bash

# AI Diagnostic Companion Startup Script
# This script starts both the Next.js frontend and Python ML service

echo "🚀 Starting AI Diagnostic Companion..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Install Node.js dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Start Python ML service in background
echo "🐍 Starting Python ML service..."
cd ml
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt > /dev/null 2>&1
python diseasePredictor.py &
ML_PID=$!
cd ..

# Wait a moment for ML service to start
sleep 3

# Start Next.js development server
echo "⚛️ Starting Next.js development server..."
echo ""
echo "🌟 Services starting:"
echo "   Frontend: http://localhost:3000"
echo "   ML Service: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $ML_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start Next.js dev server
npm run dev
