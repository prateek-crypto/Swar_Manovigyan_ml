@echo off
echo ============================================================
echo EMOTION-AWARE MUSIC RECOMMENDATION SYSTEM
echo SETUP SCRIPT FOR WINDOWS
echo ============================================================
echo.

echo Installing Python packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Creating directories...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "models" mkdir "models"
if not exist "models\baseline_models" mkdir "models\baseline_models"
if not exist "notebooks" mkdir "notebooks"

echo.
echo Testing installation...
python test_system.py

echo.
echo ============================================================
echo SETUP COMPLETED!
echo ============================================================
echo.
echo Next steps:
echo 1. Run 'python train_model.py' to train the models
echo 2. Run 'streamlit run src/frontend/app.py' to start the app
echo 3. Run 'python demo.py' to see a demonstration
echo.
pause
