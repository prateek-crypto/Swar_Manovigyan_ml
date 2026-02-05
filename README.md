# 🎵 Swar_Manovigyan_ML

> **Emotion-Aware Music Therapy System**  
> A deep learning platform for personalized music recommendations based on continuous arousal-valence emotion detection from audio signals.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-36%20passed-success.svg)](tests/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Azure OpenAI Integration](#-azure-openai-integration)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**Swar_Manovigyan_ML** bridges the gap between computational music emotion recognition and therapeutic applications. Unlike traditional popularity-based or genre-based music recommenders, this system provides:

- **Continuous emotion prediction** using arousal-valence (A/V) coordinates instead of discrete mood categories
- **Frame-level temporal modeling** with mel-spectrograms for true audio sequence analysis
- **Dual feature pathways** supporting both tabular (Spotify-style) and mel-spectrogram inputs
- **AI-powered recommendations** using GPT-4o for personalized, context-aware music suggestions
- **Therapeutic focus** with emotion-aware guidance and explanations

### Research Foundation

Built on Russell's circumplex model of affect, this system uses Bidirectional LSTM networks to predict continuous arousal and valence coordinates from audio features, enabling fine-grained emotional analysis for music therapy applications.

---

## ✨ Key Features

### Core Capabilities

- 🧠 **BiLSTM Emotion Regression**: Predicts continuous arousal-valence coordinates [0,1] from audio
- 🎼 **Dual Input Pathways**: 
  - **Tabular**: 11-dimensional Spotify-style features (energy, tempo, valence, etc.)
  - **Mel-Spectrogram**: 128-band mel-spectrograms with frame-level temporal sequences
- 📊 **Real-time Inference**: CLI and web-based audio analysis (supports WAV, MP3, FLAC, OGG)
- 🎨 **Interactive UI**: Streamlit-based web application with three input modes
- 🤖 **AI-Powered Recommendations**: GPT-4o generates personalized music style suggestions

### Advanced Features

- 🔄 **Frame-Level Temporal Modeling**: True temporal sequences from consecutive audio frames
- 📈 **Emotion Quadrant Mapping**: Maps A/V coordinates to 4 emotion classes (Sad, Calm, Stressed, Happy)
- 💡 **Therapeutic Explanations**: AI-generated insights about detected emotions and music's therapeutic role
- 🧪 **Comprehensive Testing**: 36+ unit and integration tests covering all components
- 🔧 **Production-Ready**: Feature normalization, error handling, graceful fallbacks

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Layer                               │
│  ┌──────────────┐              ┌──────────────┐            │
│  │ Tabular (11) │              │ Mel (128)    │            │
│  │ Features     │              │ Spectrogram  │            │
│  └──────┬───────┘              └──────┬───────┘            │
│         │                              │                    │
│         └──────────┬───────────────────┘                    │
│                    │                                        │
│         ┌──────────▼──────────┐                            │
│         │  Sequence Building  │                            │
│         │  (Sliding Windows)  │                            │
│         └──────────┬──────────┘                            │
└────────────────────┼──────────────────────────────────────┘
                     │
         ┌───────────▼────────────┐
         │  BiLSTM Regressor      │
         │  ┌──────────────────┐  │
         │  │ BiLSTM (96 units)│  │
         │  │ + Dropout + BN   │  │
         │  └────────┬─────────┘  │
         │  ┌────────▼─────────┐  │
         │  │ BiLSTM (48 units)│  │
         │  │ + Dropout + BN   │  │
         │  └────────┬─────────┘  │
         │  ┌────────▼─────────┐  │
         │  │ Dense (128→64)   │  │
         │  └────────┬─────────┘  │
         │  ┌────────▼─────────┐  │
         │  │ Output (2)       │  │
         │  │ [Arousal,Valence]│  │
         │  └──────────────────┘  │
         └───────────┬─────────────┘
                     │
         ┌───────────▼────────────┐
         │  Emotion Quadrant      │
         │  + GPT-4o Suggestions │
         └────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Azure OpenAI account for AI-powered features

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Swar_Manovigyan_ML

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# 1. Prepare data (if needed)
python -m src.utils.data_analysis

# 2. Train model (tabular pathway)
python -m src.train_av \
    --csv data/processed/spotify_features_with_emotions.csv \
    --epochs 20 \
    --checkpoint models/av_regressor.keras

# 3. Run inference (CLI)
python -m src.inference_av \
    --audio_path path/to/audio.wav \
    --checkpoint models/av_regressor.keras

# 4. Launch web application
streamlit run src/frontend/app.py
```

---

## 📦 Installation

### Required Dependencies

See `requirements.txt` for complete list. Core dependencies:

- **TensorFlow** ≥2.10.0 - Deep learning framework
- **Streamlit** ≥1.28.0 - Web application framework
- **Librosa** ≥0.9.0 - Audio analysis
- **Pandas** ≥1.5.0 - Data manipulation
- **NumPy** ≥1.21.0 - Numerical computing
- **Plotly** ≥5.0.0 - Interactive visualizations

### Optional Dependencies

- **Azure OpenAI** (`openai`, `python-dotenv`) - For AI-powered recommendations
- **pytest** ≥7.0.0 - For running test suite

### Platform-Specific Notes

**Windows Users:**
- May require Visual C++ Redistributables for TensorFlow
- Download: https://aka.ms/vs/17/release/vc_redist.x64.exe

**Protobuf Compatibility:**
- Uses `protobuf>=5.28,<6` for TensorFlow 2.20+ and Streamlit compatibility
- Automatically installed via `requirements.txt`

---

## 💻 Usage

### Training Models

#### Tabular Feature Model (11-dimensional)

```bash
python -m src.train_av \
    --csv data/processed/spotify_features_with_emotions.csv \
    --sequence_length 10 \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001 \
    --checkpoint models/av_regressor.keras
```

**Output:**
- `models/av_regressor.keras` - Trained model checkpoint
- `models/feature_stats_av.json` - Feature normalization statistics (for inference)

#### Mel-Spectrogram Model (128-dimensional, Frame-Level)

```bash
python -m src.train_av_mel \
    --audio_dir data/audio \
    --csv data/processed/mel_training_labels.csv \
    --filename_column filename \
    --sequence_length 10 \
    --epochs 20 \
    --checkpoint models/av_regressor_mel.keras
```

**Requirements:**
- Audio files matching filenames in CSV
- CSV with columns: `filename`, `arousal`, `enhanced_valence`

**Output:**
- `models/av_regressor_mel.keras` - Trained mel-spectrogram model
- `models/mel_stats_av.json` - Mel-spectrogram configuration

### Inference

#### Command-Line Interface

```bash
# Tabular model
python -m src.inference_av \
    --audio_path song.wav \
    --checkpoint models/av_regressor.keras

# Mel-spectrogram model
python -m src.inference_av \
    --audio_path song.wav \
    --checkpoint models/av_regressor_mel.keras
```

**Output:** JSON with `arousal` and `valence` values [0,1]

#### Web Application

```bash
streamlit run src/frontend/app.py
```

**Three Input Modes:**

1. **Manual Emotion Selection**
   - Direct arousal-valence slider input
   - Instant quadrant mapping and recommendations

2. **Audio Features Input**
   - Manual feature specification (9 sliders)
   - LSTM classification + baseline model comparison
   - A/V estimation from features

3. **Audio Upload (A/V)**
   - Real-time audio file analysis
   - Automatic feature extraction
   - A/V prediction with visualization

---

## 🧠 Model Architecture

### Preprocessing Pipeline

#### Feature Extraction

**Tabular Features (11-dim):**
- `acousticness`, `danceability`, `energy`, `instrumentalness`, `liveness`
- `loudness` (dB), `speechiness`, `tempo` (BPM), `valence`
- **Derived:** `arousal`, `enhanced_valence`

**Mel-Spectrogram Features (128-dim):**
- 128 mel bands, 22050 Hz sample rate
- Hop length: 512 samples, FFT: 2048
- Per-file min-max normalization [0,1]

#### Emotion Label Computation

**Arousal:**
```
arousal = 0.4 × energy + 0.3 × loudness_norm + 0.3 × tempo_norm
```
- `loudness_norm`: [-60, 0] dB → [0, 1]
- `tempo_norm`: [50, 200] BPM → [0, 1]

**Enhanced Valence:**
```
enhanced_valence = 0.6 × valence + 0.3 × danceability + 0.1 × (1 - acousticness)
```

#### Sequence Building

- **Tabular:** Sliding window over track sequences (T=10 timesteps)
- **Mel-Spectrogram:** Frame-level sliding window over audio frames (true temporal modeling)
- Z-score standardization for tabular features
- Padding/cropping to fixed sequence length

### Neural Network Architecture

**Input:** `(batch, sequence_length, num_features)`
- Tabular: `(batch, 10, 11)`
- Mel-Spectrogram: `(batch, 10, 128)`

**Layers:**
1. **Bidirectional LSTM** (96 units/direction, return_sequences=True)
   - Dropout (0.3) + BatchNormalization
2. **Bidirectional LSTM** (48 units/direction)
   - Dropout (0.3) + BatchNormalization
3. **Dense** (128 units, ReLU)
   - Dropout (0.3)
4. **Dense** (64 units, ReLU)
5. **Output** (2 units, Sigmoid) → `[arousal, valence]` ∈ [0,1]

**Training:**
- Optimizer: Adam (lr=1e-3)
- Loss: Mean Squared Error (MSE)
- Callbacks: EarlyStopping, ReduceLROnPlateau
- Data Split: 60/20/20 (train/val/test)

---

## 🤖 Azure OpenAI Integration

### Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Configure Azure OpenAI credentials in `.env`:
   ```env
   AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com
   AZURE_OPENAI_API_KEY=YOUR_API_KEY
   AZURE_OPENAI_DEPLOYMENT=gpt-4o
   AZURE_OPENAI_API_VERSION=2025-01-01-preview
   ```

3. **Important:** Never commit `.env` (already in `.gitignore`)

### Features Enabled

When Azure OpenAI is configured, the system provides:

- **AI-Generated Music Styles**: GPT-4o creates personalized music style/playlist suggestions based on detected emotion and A/V coordinates
- **Therapeutic Explanations**: Context-aware insights about detected emotions and music's therapeutic role
- **Recommendation Blurbs**: AI-generated explanations tying music suggestions to emotional state
- **Sample Track Suggestions**: Fictional example tracks matching emotional context

**Fallback Behavior:** If Azure OpenAI is unavailable, the system gracefully falls back to hardcoded recommendations.

---

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_audio_features.py -v

# Run with coverage (requires pytest-cov)
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

**36 tests** covering:
- ✅ Audio feature extraction (5 tests)
- ✅ Azure OpenAI integration (11 tests)
- ✅ Data analysis & emotion labeling (4 tests)
- ✅ Inference pipeline (4 tests)
- ✅ Model architecture (6 tests)
- ✅ Sequence building (6 tests)

**Test Results:** See `docs/TEST_RESULTS.md` for detailed results.

### Known Issues

- **Windows TensorFlow DLL:** Some Windows systems may require Visual C++ Redistributables. Tests pass under pytest; Streamlit app works normally.
- **Protobuf Compatibility:** Resolved via `protobuf>=5.28,<6` in requirements.txt

---

## 📁 Project Structure

```
Swar_Manovigyan_ML/
├── src/
│   ├── frontend/
│   │   └── app.py                    # Streamlit web application
│   ├── models/
│   │   ├── av_regressor.py           # BiLSTM A/V regressor
│   │   ├── baseline_models.py         # Baseline ML models
│   │   ├── lstm_model.py             # 4-class LSTM classifier
│   │   └── training_pipeline.py      # Classification training
│   ├── utils/
│   │   ├── audio_features.py         # Feature extraction (mel + tabular)
│   │   ├── azure_openai_service.py   # Azure OpenAI integration
│   │   ├── data_analysis.py          # Emotion labeling & preprocessing
│   │   └── feature_stats.py          # Normalization utilities
│   ├── train_av.py                   # Tabular model training
│   ├── train_av_mel.py               # Mel-spectrogram model training
│   └── inference_av.py               # CLI inference script
├── data/
│   ├── audio/                        # Audio files for mel training
│   ├── processed/                    # Processed CSVs with emotion labels
│   └── raw/                          # Raw Spotify features CSV
├── models/                           # Trained model checkpoints
├── scripts/
│   └── create_synthetic_audio.py     # Synthetic audio generation
├── tests/                            # Test suite (36 tests)
│   ├── test_audio_features.py
│   ├── test_azure_openai.py
│   ├── test_data_analysis.py
│   ├── test_inference.py
│   ├── test_models.py
│   └── test_sequence_building.py
├── docs/                             # Documentation
│   ├── EVALUATION_REPORT.md          # Architecture evaluation
│   ├── IMPLEMENTATION_SUMMARY.md     # Implementation details
│   ├── TEST_RESULTS.md               # Test results summary
│   ├── TEST_SUMMARY.md               # Testing guide
│   ├── TODO_SDLC.md                  # Development roadmap
│   ├── info.txt                      # Project review information
│   └── submission/                   # Submission documents
│       ├── Document 1 - Submission of Project Details.docx
│       ├── Document 2 - Submission of Student Log Book.docx
│       ├── PPT Template.pptx
│       ├── Logbook.pdf
│       ├── ML-report (1).pdf
│       ├── ML-report.docx
│       └── Swar_Manovigyan (1).pdf
├── .env.example                      # Environment variable template
├── requirements.txt                  # Python dependencies
├── pytest.ini                        # Pytest configuration
└── README.md                         # This file
```

---

## 🔬 Research & Methodology

### Theoretical Foundation

- **Emotion Model:** Russell's Circumplex Model of Affect (arousal-valence space)
- **Deep Learning:** Bidirectional LSTM for temporal sequence modeling
- **Feature Engineering:** Dual pathway (tabular + mel-spectrogram) for flexibility

### Key Innovations

1. **Frame-Level Temporal Modeling:** True temporal sequences from consecutive audio frames (not tiled vectors)
2. **Continuous Emotion Prediction:** Arousal-valence regression instead of discrete classification
3. **Therapeutic Focus:** Emotion-aware recommendations with AI-generated explanations
4. **Dual Pathway Support:** Both Spotify-style features and raw audio mel-spectrograms

### Performance Metrics

- **Test MSE:** ~4×10⁻⁴ (tabular model)
- **Test MAE:** ~0.016 (arousal-valence prediction)
- **Validation MAE:** ~0.016 (best epoch)

---

## 🚧 Future Work

- [ ] User-specific fine-tuning on personalized A/V data
- [ ] Multi-modal integration (physiological signals: heart rate, skin conductance)
- [ ] Attention mechanisms over temporal frames for interpretability
- [ ] Ensemble models (tabular + mel-spectrogram)
- [ ] Real-time streaming audio analysis
- [ ] Integration with music streaming APIs (Spotify, Apple Music)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

See `docs/TODO_SDLC.md` for structured development checklist.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

**Swar_Manovigyan_ML Team**
- Adit Jain
- Mehir Singh
- Ayush Pandey

---

## 🙏 Acknowledgments

- **Research References:** See `docs/EVALUATION_REPORT.md` for citations
- **Libraries:** TensorFlow, Librosa, Streamlit, Plotly
- **Data:** Spotify Features Dataset

---

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check `docs/EVALUATION_REPORT.md` for known issues and solutions
- Review `docs/TEST_SUMMARY.md` for testing guidance

---

**Built with ❤️ for computational music therapy research**
