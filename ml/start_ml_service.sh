#!/bin/bash

# Medical Report ML Service Startup Script
# This script sets up and starts the Python Flask ML service

echo "🚀 Starting Medical Report ML Service..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3."
    exit 1
fi

# Navigate to ML directory
cd ml

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Start the Flask service
echo "🌟 Starting Flask ML service on http://localhost:5000"
echo "📊 Available endpoints:"
echo "   GET  /health - Health check"
echo "   POST /predict - Disease prediction"
echo "   GET  /models - List available patterns"
echo "   POST /analyze - Analyze metrics"
echo ""
echo "Press Ctrl+C to stop the service"

python diseasePredictor.py
