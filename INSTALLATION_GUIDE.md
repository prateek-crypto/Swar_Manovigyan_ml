# 🚀 Installation Guide

## Quick Installation

### Option 1: Automated Setup (Recommended)
```bash
python setup.py
```

### Option 2: Manual Installation
```bash
# Install core packages first
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly

# Install TensorFlow (optional, for LSTM models)
pip install tensorflow

# Install additional packages
pip install librosa soundfile joblib tqdm
```

### Option 3: Windows Batch File
```bash
setup.bat
```

## Troubleshooting

### Common Issues

#### 1. **Pandas Installation Error**
If you get pandas installation errors:
```bash
# Try installing pandas separately
pip install pandas --no-cache-dir

# Or use conda instead
conda install pandas numpy scikit-learn matplotlib seaborn
```

#### 2. **TensorFlow Installation Error**
If TensorFlow fails to install:
```bash
# Install CPU-only version
pip install tensorflow-cpu

# Or skip TensorFlow for now (baseline models will still work)
```

#### 3. **Permission Errors**
If you get permission errors:
```bash
# Use user installation
pip install --user -r requirements.txt

# Or create a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

#### 4. **Directory Creation Error**
If you get directory creation errors:
```bash
# Manually create directories
mkdir data\raw data\processed models models\baseline_models notebooks
```

## Minimal Installation

If you want to run just the basic functionality:

```bash
# Install only essential packages
pip install streamlit pandas numpy scikit-learn matplotlib

# Run the app (baseline models only)
streamlit run src/frontend/app.py
```

## Verification

After installation, test the system:

```bash
# Test basic functionality
python test_system.py

# Run a quick demo
python demo.py
```

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **OS**: Windows, macOS, or Linux

## Getting Help

If you encounter issues:

1. Check the error messages carefully
2. Try the troubleshooting steps above
3. Install packages one by one to identify the problematic package
4. Use virtual environments to avoid conflicts
5. Check the [GitHub Issues](https://github.com/Adit-Jain-srm/Swar_Manovigyan_ML/issues) page

## Success Indicators

You'll know the installation is successful when:

- ✅ `python test_system.py` runs without errors
- ✅ `streamlit run src/frontend/app.py` starts the web app
- ✅ You can access the app at `http://localhost:8501`
- ✅ The emotion prediction interface loads correctly

---

**Need help?** Check the README.md or create an issue on GitHub!
