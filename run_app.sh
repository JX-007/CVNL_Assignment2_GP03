#!/bin/bash

echo "========================================"
echo " Changi Ops Assistant - Quick Start"
echo "========================================"
echo ""

# Check setup first
echo "Step 1: Checking setup..."
python3 check_setup.py
echo ""

# Ask user if they want to continue
echo ""
read -p "Continue to run Streamlit app? (y/n): " continue
if [ "$continue" != "y" ] && [ "$continue" != "Y" ]; then
    echo "Exiting..."
    exit 0
fi

# Run Streamlit
echo ""
echo "Step 2: Starting Streamlit app..."
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run changi_ops_assistant.py
