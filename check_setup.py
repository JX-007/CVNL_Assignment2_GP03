"""
Quick Setup Script for Changi Ops Assistant

This script helps you verify that everything is set up correctly
before running the Streamlit app.

Run this first to check your setup!
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a required file exists"""
    if os.path.exists(filepath):
        print(f"{description}: Found")
        return True
    else:
        print(f"{description}: NOT FOUND")
        print(f"   Expected location: {filepath}")
        return False

def check_imports():
    """Check if required packages are installed"""
    print("\nChecking Python Packages...")
    print("-" * 50)
    
    packages = {
        'streamlit': 'Streamlit web framework',
        'torch': 'PyTorch (deep learning)',
        'numpy': 'NumPy (numerical computing)',
        'PIL': 'Pillow (image processing)',
    }
    
    all_ok = True
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"{package}: Installed ({description})")
        except ImportError:
            print(f"{package}: NOT INSTALLED ({description})")
            print(f"   Install with: pip install {package}")
            all_ok = False
    
    return all_ok

def check_model_files():
    """Check if model files exist"""
    print("\nChecking Model Files...")
    print("-" * 50)
    
    files = {
        'models/rnn_intent/checkpoints/intent_best.pth': 'RNN Model Checkpoint',
        'models/rnn_intent/data/intent_vocab.json': 'Vocabulary File',
    }
    
    all_ok = True
    for filepath, description in files.items():
        if not check_file_exists(filepath, description):
            all_ok = False
    
    return all_ok

def main():
    print("=" * 60)
    print("CHANGI OPS ASSISTANT - SETUP VERIFICATION")
    print("=" * 60)
    
    # Check Python version
    print("\nPython Version:")
    print("-" * 50)
    python_version = sys.version.split()[0]
    print(f"Python {python_version}")
    
    if sys.version_info < (3, 8):
        print("Warning: Python 3.8+ recommended")
    
    # Check packages
    packages_ok = check_imports()
    
    # Check model files
    files_ok = check_model_files()
    
    # Final summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    if packages_ok and files_ok:
        print("All checks passed! You're ready to run the app!")
        print("\nNext Steps:")
        print("   1. Open terminal in this directory")
        print("   2. Run: streamlit run changi_ops_assistant.py")
        print("   3. Browser will open automatically at http://localhost:8501")
        print("\nGood luck with your demo!")
    else:
        print("Some requirements are missing. Please fix the issues above.")
        
        if not packages_ok:
            print("\nTo install missing packages:")
            print("   pip install -r requirements.txt")
        
        if not files_ok:
            print("\nTo generate missing model files:")
            print("   1. Open your Jupyter notebook (rnn_intent.ipynb)")
            print("   2. Run all cells (especially the last cell that saves the model)")
            print("   3. Verify that checkpoints/ and data/ folders are created")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
