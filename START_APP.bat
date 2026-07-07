@echo off
title House Price Predictor - Starting...
color 0A

echo.
echo  ============================================
echo    House Price Predictor - Starting...
echo  ============================================
echo.

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Check if model exists, if not train it
if not exist "src\model.pkl" (
    echo  [!] Model not found. Running training pipeline...
    set PYTHONUTF8=1
    python run_pipeline.py --full
    echo.
)

echo  [OK] Starting web server...
echo  [OK] Opening browser in 3 seconds...
echo.
echo  Your app will open at: http://localhost:5000
echo  Press Ctrl+C to stop the server.
echo.

:: Open browser after 3 second delay (in background)
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"

:: Start Flask app
set PYTHONUTF8=1
python app.py

pause
