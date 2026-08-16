@echo off
REM Production startup script for ParkVision (Windows)

echo =========================================
echo ParkVision - Production Startup
echo =========================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check if model exists
if not exist "models\best.pt" (
    echo ERROR: Model not found at models\best.pt
    echo Please train the model first: python scripts\train_model.py
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo WARNING: .env file not found. Using default configuration.
    echo Copy .env.example to .env and configure for production.
)

REM Start the application
echo Starting ParkVision...
echo Access at: http://localhost:5000
echo Press Ctrl+C to stop
echo =========================================

python src\app.py
