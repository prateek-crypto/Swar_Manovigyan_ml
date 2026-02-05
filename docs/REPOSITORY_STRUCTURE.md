# Repository Structure

This document describes the organization of the Swar_Manovigyan_ML repository.

## 📁 Directory Organization

### Root Directory

The root directory contains only **essential files** needed to understand and run the project:

```
Swar_Manovigyan_ML/
├── README.md              # Main project documentation
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
├── .gitignore            # Git ignore rules
├── .gitattributes        # Git line ending normalization
├── pytest.ini            # Pytest configuration
├── src/                  # Source code
├── data/                 # Data files
├── models/               # Trained model checkpoints
├── tests/                # Test suite
├── scripts/              # Utility scripts
└── docs/                 # Documentation and reports
```

### Source Code (`src/`)

- **`frontend/`** - Streamlit web application
- **`models/`** - Model definitions (BiLSTM, baselines, LSTM classifier)
- **`utils/`** - Utility modules (feature extraction, data analysis, Azure OpenAI)
- **`train_av.py`** - Tabular model training script
- **`train_av_mel.py`** - Mel-spectrogram model training script
- **`inference_av.py`** - CLI inference script

### Data (`data/`)

- **`raw/`** - Raw datasets (Spotify features CSV)
- **`processed/`** - Processed datasets with emotion labels
- **`audio/`** - Audio files for mel-spectrogram training

### Models (`models/`)

- Trained model checkpoints (`.keras` files)
- Feature statistics JSON files
- Baseline model files (`.joblib`)

### Tests (`tests/`)

- **36 test files** covering all components
- Test configuration in `conftest.py`
- Pytest markers for dependency management

### Documentation (`docs/`)

#### Technical Documentation

- **`EVALUATION_REPORT.md`** - Architecture evaluation and issue analysis
- **`IMPLEMENTATION_SUMMARY.md`** - Implementation details for mel-spectrogram and testing
- **`TEST_RESULTS.md`** - Complete test suite results
- **`TEST_SUMMARY.md`** - Testing guide and troubleshooting
- **`TODO_SDLC.md`** - Development lifecycle checklist
- **`REPOSITORY_STRUCTURE.md`** - This file

#### Submission Materials (`docs/submission/`)

Academic submission documents (not required for running the project):

- Project details and logbook templates
- Presentation templates
- Project reports (PDF/DOCX)

### Scripts (`scripts/`)

- **`create_synthetic_audio.py`** - Generate synthetic WAV files for testing

## 🎯 Design Principles

### Clean Root Directory

Only essential files are kept in the root:
- **README.md** - First thing users see
- **requirements.txt** - Dependencies
- **Configuration files** - `.env.example`, `pytest.ini`, `.gitignore`

### Organized Documentation

All documentation is centralized in `docs/`:
- Technical docs are easily accessible
- Submission materials are separated
- Clear navigation via `docs/README.md`

### Separation of Concerns

- **Source code** (`src/`) - All application logic
- **Data** (`data/`) - All datasets
- **Models** (`models/`) - All trained models
- **Tests** (`tests/`) - All test code
- **Documentation** (`docs/`) - All documentation

## 📝 File Naming Conventions

- **Python files**: `snake_case.py`
- **Markdown files**: `UPPER_SNAKE_CASE.md` or `Title Case.md`
- **Data files**: Descriptive names with extensions
- **Model files**: Descriptive names with `.keras` extension

## 🔍 Finding Files

### Quick Reference

- **Want to run the app?** → `src/frontend/app.py`
- **Want to train a model?** → `src/train_av.py` or `src/train_av_mel.py`
- **Want to understand architecture?** → `docs/EVALUATION_REPORT.md`
- **Want to see test results?** → `docs/TEST_RESULTS.md`
- **Want to contribute?** → `docs/TODO_SDLC.md`

### Common Paths

```bash
# Main application
streamlit run src/frontend/app.py

# Training
python -m src.train_av --csv data/processed/spotify_features_with_emotions.csv

# Testing
pytest tests/ -v

# Documentation
docs/README.md
```

---

**Last Updated:** 2025-02-05  
**Maintained by:** Swar_Manovigyan_ML Team
