@echo off
echo ========================================
echo  Changi Ops Assistant - Quick Start
echo ========================================
echo.

REM Check setup first
echo Step 1: Checking setup...
python check_setup.py
echo.

REM Ask user if they want to continue
echo.
set /p continue="Continue to run Streamlit app? (y/n): "
if /i "%continue%" NEQ "y" (
    echo Exiting...
    pause
    exit
)

REM Run Streamlit
echo.
echo Step 2: Starting Streamlit app...
echo.
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

streamlit run changi_ops_assistant.py

pause
