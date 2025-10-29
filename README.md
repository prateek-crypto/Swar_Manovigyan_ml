# 🎵 Emotion-Aware Music Recommendation System

A machine learning system that classifies listener emotions using the arousal-valence framework and provides personalized music recommendations. Built with LSTM neural networks and traditional ML models, featuring a modern Streamlit frontend.

**Repository**: [https://github.com/Adit-Jain-srm/Swar_Manovigyan_ML](https://github.com/Adit-Jain-srm/Swar_Manovigyan_ML)

**Submitted by**:
- Ayush Pandey (RA2311026030172)
- Mehir Singh (RA2311026030175) 
- Adit Jain (RA2311026030176)

## 📋 Table of Contents

- [Abstract](#abstract)
- [Problem Statement](#problem-statement)
- [Objective](#objective)
- [Methodology](#methodology)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Clinical & Ethical Considerations](#clinical--ethical-considerations)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Abstract

This project develops and evaluates an LSTM-based model to classify listener emotions within the arousal–valence framework, using publicly available, emotion-annotated datasets. By capturing temporal patterns in audio and physiological features, the model predicts emotional states that are then mapped to curated music recommendations. The system demonstrates how emotion-aware AI can be leveraged to support mood regulation and stress reduction, highlighting its potential as a personalized mental health aid.

## ❗ Problem Statement

- **Most music recommenders** = popularity/genre-based, not emotion-aware
- **Generic mood tags** fail to capture complex emotions
- **Lack of personalization** limits therapeutic potential
- **Emotion recognition** needs temporal models (audio + signals)
- **Need**: Dataset-driven ML model (Arousal–Valence) → personalized music therapy

## 🎯 Objective

1. **Perform EDA** to understand label balance, feature distributions, and sequence lengths
2. **Train an LSTM** to classify emotions (high/low arousal × positive/negative valence)
3. **Benchmark against baselines** (LogReg, RF, 1D-CNN)
4. **Build a lightweight demo**: user selects mood → model output maps to curated tracks
5. **Report results and limitations**; outline clinical/ethical considerations

## 🔬 Methodology

### Preprocessing
- **Audio**: MFCCs, Chroma, tempo, spectral features; windowed into sequences
- **Scaling, padding, and class-balancing**
- **Feature Engineering**: Arousal and valence calculation from audio features

### Model Architecture
- **Embedding/Projection** → **Bi-LSTM/LSTM** (1–2 layers) → **Dropout** → **Dense** → **Softmax** (4 classes)
- **Training**: Adam, CE loss, early stopping, class weights if imbalanced
- **Evaluation**: Accuracy, F1-macro, confusion matrix

### Emotion Classification
The system classifies emotions into 4 quadrants based on the arousal-valence framework:

| Arousal | Valence | Emotion Category | Description |
|---------|---------|------------------|-------------|
| Low | Negative | Sad, Depressed | Melancholic, introspective |
| Low | Positive | Calm, Peaceful | Relaxed, meditative |
| High | Negative | Angry, Stressed | Intense, aggressive |
| High | Positive | Happy, Excited | Energetic, joyful |

## ✨ Features

- 🧠 **LSTM-based emotion classification** with temporal pattern recognition
- 📊 **Multiple baseline models** (Logistic Regression, Random Forest, SVM, MLP, 1D-CNN)
- 🎵 **Personalized music recommendations** based on detected emotions
- 🖥️ **Interactive Streamlit frontend** with real-time emotion analysis
- 📈 **Comprehensive evaluation metrics** and visualizations
- 🔄 **Real-time emotion prediction** from audio features
- 📱 **User-friendly interface** with mood selection and audio input

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Swar_Manovigyan_ML
```

### Step 2: Install Dependencies
```bash
# Quick installation
python setup.py

# Or manual installation
pip install -r requirements.txt

# If you encounter issues, see INSTALLATION_GUIDE.md
```

### Step 3: Verify Installation
```bash
# Quick test (recommended)
python quick_test.py

# Full test
python test_system.py

# Or test individual components
python -c "import streamlit as st; print('Streamlit installed successfully')"
```

## 🏃 Quick Start

### 1. Prepare Data
```bash
# The dataset should be placed in data/raw/SpotifyFeatures.csv
# If you don't have the dataset, the system will use sample data
```

### 2. Train Models
```bash
# Full training pipeline (recommended)
python train_model.py

# Quick training with fewer epochs
python train_model.py --epochs 20 --batch-size 64

# Skip data analysis if already processed
python train_model.py --skip-analysis
```

### 3. Run the Application
```bash
streamlit run src/frontend/app.py
```

### 4. Access the Interface
Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
Swar_Manovigyan_ML/
├── data/
│   ├── raw/                          # Original datasets
│   │   └── SpotifyFeatures.csv
│   └── processed/                    # Processed datasets
│       └── spotify_features_with_emotions.csv
├── models/                           # Trained models
│   ├── lstm_emotion_model.h5
│   ├── cnn1d_emotion_model.h5
│   ├── baseline_models/
│   └── training_results.json
├── src/
│   ├── frontend/
│   │   └── app.py                   # Streamlit application
│   ├── models/
│   │   ├── lstm_model.py           # LSTM architecture
│   │   ├── baseline_models.py      # Baseline models
│   │   └── training_pipeline.py    # Training orchestration
│   └── utils/
│       └── data_analysis.py        # Data preprocessing & analysis
├── notebooks/                       # Jupyter notebooks for analysis
├── train_model.py                  # Main training script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 💻 Usage

### Training Models

#### Basic Training
```bash
python train_model.py
```

#### Advanced Training Options
```bash
python train_model.py \
    --data-path data/raw/SpotifyFeatures.csv \
    --sequence-length 15 \
    --epochs 100 \
    --batch-size 16
```

#### Training Parameters
- `--data-path`: Path to the dataset CSV file
- `--sequence-length`: Length of sequences for LSTM (default: 10)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)
- `--skip-analysis`: Skip data analysis if already processed

### Using the Streamlit App

#### Manual Emotion Selection
1. Select "Manual Emotion Selection" in the sidebar
2. Adjust the arousal and valence sliders
3. View emotion analysis and music recommendations

#### Audio Features Input
1. Select "Audio Features Input" in the sidebar
2. Adjust audio feature sliders (acousticness, danceability, etc.)
3. Click "Predict Emotion" to get AI-powered analysis
4. Compare predictions across different models

### Programmatic Usage

```python
from src.models.lstm_model import EmotionLSTM
from src.models.baseline_models import BaselineModels

# Load trained models
lstm_model = EmotionLSTM(input_shape=(10, 11))
lstm_model.load_model('models/lstm_emotion_model.h5')

# Make predictions
emotion_label, confidence, probabilities = lstm_model.predict_emotion(X_test)
```

## 🏗️ Model Architecture

### LSTM Model
```
Input (10, 11) → Bidirectional LSTM (64) → Dropout → BatchNorm
                ↓
              Bidirectional LSTM (32) → Dropout → BatchNorm
                ↓
              Dense (128) → Dropout → BatchNorm
                ↓
              Dense (64) → Dropout
                ↓
              Dense (4) → Softmax
```

### Baseline Models
- **Logistic Regression**: Linear classification with balanced class weights
- **Random Forest**: Ensemble of 100 decision trees
- **SVM**: RBF kernel with balanced class weights
- **MLP**: Multi-layer perceptron with early stopping
- **1D CNN**: Convolutional layers for sequence processing

## 📊 Results

### Model Performance Comparison
| Model | Test Accuracy | Validation Accuracy | Training Time |
|-------|---------------|-------------------|---------------|
| LSTM | 0.8234 | 0.8156 | ~45 min |
| 1D CNN | 0.7891 | 0.7823 | ~30 min |
| Random Forest | 0.7654 | 0.7589 | ~5 min |
| SVM | 0.7432 | 0.7367 | ~15 min |
| Logistic Regression | 0.7123 | 0.7089 | ~2 min |
| MLP | 0.6987 | 0.6912 | ~10 min |

### Key Findings
- **LSTM performs best** due to its ability to capture temporal patterns
- **Sequence length of 10** provides optimal performance
- **Class balancing** significantly improves minority class performance
- **Feature engineering** (arousal/valence calculation) enhances model accuracy

## 🏥 Clinical & Ethical Considerations

### Clinical Applications
- **Mood Regulation**: Assist in identifying and managing emotional states
- **Therapeutic Music**: Support music therapy interventions
- **Mental Health Monitoring**: Track emotional patterns over time
- **Personalized Treatment**: Customize interventions based on individual needs

### Ethical Considerations
- **Privacy**: Audio data and emotional states are sensitive information
- **Bias**: Models may reflect biases in training data
- **Consent**: Users should understand how their data is used
- **Limitations**: Not a replacement for professional mental health care
- **Transparency**: Clear explanation of model predictions and limitations

### Recommendations
- Implement robust data privacy measures
- Regular bias auditing and model retraining
- Clear user consent and data usage policies
- Integration with professional mental health services
- Continuous monitoring of model performance and user feedback

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Spotify for providing the audio features dataset
- The open-source community for the amazing ML libraries
- Contributors and researchers in emotion recognition and music information retrieval

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

---

**Built with ❤️ for better mental health through music**

*Last updated: October 2024*