# Test Suite Summary

## Overview

Comprehensive test suite for Swar_Manovigyan_ML covering:
- Audio feature extraction (tabular and mel-spectrogram)
- Sequence building (tabular and frame-level temporal)
- Model architecture and inference
- Data analysis and emotion labeling
- End-to-end inference pipeline

## Test Files

### `tests/test_audio_features.py`
Tests for audio feature extraction:
- ✅ Mel-spectrogram extraction with correct shape and normalization
- ✅ Tabular feature extraction with correct shape and ranges
- ✅ Handling of short audio (padding)
- ✅ Sample rate conversion
- ✅ Feature consistency (tiling for tabular)

**Dependencies:** `soundfile`, `librosa`

### `tests/test_sequence_building.py`
Tests for sequence building:
- ✅ Tabular sequence building from DataFrame
- ✅ Frame-level sequence building from mel-spectrograms
- ✅ Stride handling
- ✅ Sequence limits
- ✅ Short audio padding

**Dependencies:** `pandas`, `numpy` (TensorFlow optional for imports)

### `tests/test_models.py`
Tests for model architecture:
- ✅ AV regressor build for tabular input (10, 11)
- ✅ AV regressor build for mel input (10, 128)
- ✅ Model prediction and output bounds [0,1]
- ✅ Model save/load consistency
- ✅ Model evaluation metrics

**Dependencies:** `tensorflow`, `numpy`

### `tests/test_inference.py`
Tests for inference pipeline:
- ✅ Feature stats save/load
- ✅ Z-score normalization
- ✅ End-to-end inference with tabular model
- ✅ End-to-end inference with mel-spectrogram model

**Dependencies:** `tensorflow`, `soundfile`

### `tests/test_data_analysis.py`
Tests for data analysis:
- ✅ Emotion label creation
- ✅ Fixed loudness/tempo ranges
- ✅ Quadrant mapping correctness
- ✅ Distribution analysis

**Dependencies:** `pandas`, `numpy`

## Running Tests

### Basic run:
```bash
pytest tests/ -v
```

### Run specific test file:
```bash
pytest tests/test_audio_features.py -v
```

### Run tests without TensorFlow (skip TF-dependent tests):
```bash
pytest tests/ -v -m "not tensorflow"
```

### Run tests without audio dependencies:
```bash
pytest tests/ -v -m "not audio"
```

## Known Issues

### Protobuf / TensorFlow compatibility
If you see:
```
ImportError: cannot import name 'runtime_version' from 'google.protobuf'
```

**Solution:** Use a protobuf version compatible with your TensorFlow and Streamlit:
```bash
pip install "protobuf>=5.28,<6"
```
(TensorFlow 2.20+ needs protobuf>=5.28; Streamlit prefers protobuf<6.)

### Windows TensorFlow DLL Error
If you see:
```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal
```

**Solutions:**
1. Install Visual C++ Redistributables: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Reinstall TensorFlow: `pip uninstall tensorflow && pip install tensorflow`
3. Use TensorFlow CPU-only version if GPU is not needed
4. Run tests that don't require TensorFlow: `pytest tests/ -v -m "not tensorflow"`
5. Note: The full test suite (including TensorFlow) may pass under pytest even when running `train_av_mel` or `train_av` directly hits this DLL error; fixing the DLL (e.g. VC++ redistributable) resolves both.

### Missing soundfile
If you see:
```
ModuleNotFoundError: No module named 'soundfile'
```

**Solution:**
```bash
pip install soundfile
```

## Test Coverage

Current test coverage includes:
- ✅ Core feature extraction logic
- ✅ Sequence building (both pathways)
- ✅ Model I/O and prediction
- ✅ Data preprocessing
- ✅ Inference pipeline

**Future additions:**
- Integration tests with real audio files
- Performance benchmarks
- Edge case handling (very long/short audio, corrupted files)
- Model comparison tests (tabular vs mel)

## Continuous Integration

To set up CI/CD:
1. Create `.github/workflows/test.yml` (GitHub Actions) or equivalent
2. Install dependencies: `pip install -r requirements.txt pytest pytest-cov`
3. Run: `pytest tests/ --cov=src --cov-report=xml`
4. Upload coverage reports

## Notes

- Tests use synthetic data where possible to avoid large file dependencies
- Some tests require minimal training (1 epoch) to initialize model weights
- Tests are designed to run quickly (< 1 minute total)
- All tests use fixed random seeds for reproducibility
