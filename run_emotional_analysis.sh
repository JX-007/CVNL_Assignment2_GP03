#!/bin/bash

echo "========================================"
echo " Emotional Analysis - Streamlit App"
echo "========================================"
echo ""

# Check setup first
echo "Checking setup..."
python3 check_setup.py
echo ""

# Ask user if they want to continue
echo ""
read -p "Continue to run Emotional Analysis app? (y/n): " continue
if [ "$continue" != "y" ] && [ "$continue" != "Y" ]; then
    echo "Exiting..."
    exit 0
fi

# Run Streamlit
echo ""
echo "Starting Streamlit app for Emotional Analysis..."
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run emotional_analysis_app.py
