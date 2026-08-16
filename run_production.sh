#!/bin/bash
# Production startup script for ParkVision

echo "========================================="
echo "ParkVision - Production Startup"
echo "========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if model exists
if [ ! -f "models/best.pt" ]; then
    echo "ERROR: Model not found at models/best.pt"
    echo "Please train the model first: python scripts/train_model.py"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found. Using default configuration."
    echo "Copy .env.example to .env and configure for production."
fi

# Start the application
echo "Starting ParkVision..."
echo "Access at: http://localhost:5000"
echo "Press Ctrl+C to stop"
echo "========================================="

python src/app.py
