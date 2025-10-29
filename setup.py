"""
Setup script for the Emotion-Aware Music Recommendation System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[OK] Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    directories = [
        'data/raw',
        'data/processed', 
        'models',
        'models/baseline_models',
        'notebooks'
    ]
    
    for directory in directories:
        try:
            # Check if it exists and what type it is
            if os.path.exists(directory):
                if os.path.isfile(directory):
                    print(f"Removing existing file: {directory}")
                    os.remove(directory)
                    os.makedirs(directory, exist_ok=True)
                    print(f"[OK] Created directory: {directory}")
                elif os.path.isdir(directory):
                    print(f"[OK] Directory already exists: {directory}")
                else:
                    print(f"[WARNING] {directory} exists but is neither file nor directory")
            else:
                # Create directory
                os.makedirs(directory, exist_ok=True)
                print(f"[OK] Created directory: {directory}")
        except Exception as e:
            print(f"[WARNING] Could not create {directory}: {e}")
            # Try to continue with other directories
            continue
    
    return True

def test_installation():
    """Test if the installation works"""
    print("Testing installation...")
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import streamlit as st
        import sklearn
        print("[OK] All basic packages imported successfully!")
        
        # Test TensorFlow (may fail on some systems)
        try:
            import tensorflow as tf
            print(f"[OK] TensorFlow {tf.__version__} imported successfully!")
        except ImportError as e:
            print(f"[WARNING] TensorFlow import failed: {e}")
            print("  This is common on some systems. The app will work without TensorFlow for baseline models.")
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("EMOTION-AWARE MUSIC RECOMMENDATION SYSTEM")
    print("SETUP SCRIPT")
    print("="*60)
    
    # Step 1: Create directories
    if not create_directories():
        print("[FAIL] Failed to create directories. Exiting.")
        return False
    
    # Step 2: Install requirements
    if not install_requirements():
        print("[FAIL] Failed to install requirements. Please install manually:")
        print("pip install -r requirements.txt")
        return False
    
    # Step 3: Test installation
    if not test_installation():
        print("[FAIL] Installation test failed. Please check the errors above.")
        return False
    
    print("\n" + "="*60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run 'python test_system.py' to test the system")
    print("2. Run 'python train_model.py' to train the models")
    print("3. Run 'streamlit run src/frontend/app.py' to start the app")
    print("4. Run 'python demo.py' to see a demonstration")
    print("\nFor more information, see README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
