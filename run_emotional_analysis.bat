@echo off
REM Run Emotional Analysis Streamlit App

echo.
echo ========================================
echo  Emotional Analysis - Streamlit App
echo ========================================
echo.

echo Checking setup...
python check_setup.py

echo.
echo Starting Streamlit app for Emotional Analysis...
echo.
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

streamlit run emotional_analysis_app.py

pause
