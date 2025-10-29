# 🎵 Emotion-Aware Music Recommendation System - Project Summary

**Repository**: [https://github.com/Adit-Jain-srm/Swar_Manovigyan_ML](https://github.com/Adit-Jain-srm/Swar_Manovigyan_ML)

**Submitted by**:
- Ayush Pandey (RA2311026030172)
- Mehir Singh (RA2311026030175) 
- Adit Jain (RA2311026030176)

## 🎯 Project Overview

We have successfully created a comprehensive **Emotion-Aware Music Recommendation System** that uses LSTM neural networks and traditional ML models to classify listener emotions based on the arousal-valence framework and provide personalized music recommendations.

## ✅ What Has Been Built

### 1. **Complete Project Structure**
```
Swar_Manovigyan_ML/
├── data/
│   ├── raw/SpotifyFeatures.csv          # 232,725 music tracks with audio features
│   └── processed/                       # Processed data with emotion labels
├── models/                              # Trained model storage
├── src/
│   ├── frontend/app.py                  # Streamlit web application
│   ├── models/                          # ML model implementations
│   │   ├── lstm_model.py               # LSTM architecture
│   │   ├── baseline_models.py          # Traditional ML models
│   │   └── training_pipeline.py        # Training orchestration
│   └── utils/data_analysis.py          # Data preprocessing & analysis
├── train_model.py                       # Main training script
├── demo.py                             # Demonstration script
├── test_system.py                      # System testing
├── config.py                           # Configuration settings
├── requirements.txt                    # Dependencies
└── README.md                           # Comprehensive documentation
```

### 2. **Core Features Implemented**

#### 🧠 **Machine Learning Models**
- **LSTM Model**: Bidirectional LSTM with 2 layers for temporal pattern recognition
- **1D CNN**: Convolutional neural network for sequence processing
- **Baseline Models**: Logistic Regression, Random Forest, SVM, MLP
- **Emotion Classification**: 4-class system based on arousal-valence framework

#### 🎵 **Emotion Classification System**
- **Low Arousal, Negative Valence**: Sad, Depressed (32.69% of data)
- **Low Arousal, Positive Valence**: Calm, Peaceful (17.31% of data)
- **High Arousal, Negative Valence**: Angry, Stressed (17.32% of data)
- **High Arousal, Positive Valence**: Happy, Excited (32.68% of data)

#### 🖥️ **Streamlit Frontend**
- **Manual Emotion Selection**: Interactive sliders for arousal and valence
- **Audio Features Input**: Real-time emotion prediction from audio features
- **Music Recommendations**: Personalized suggestions based on detected emotions
- **Model Comparison**: Side-by-side comparison of different models
- **Interactive Visualizations**: Arousal-valence space plotting

#### 📊 **Data Processing Pipeline**
- **Feature Engineering**: Arousal and valence calculation from audio features
- **Sequence Creation**: Time-series data preparation for LSTM
- **Class Balancing**: Handles imbalanced emotion distribution
- **Data Visualization**: Comprehensive EDA and analysis plots

### 3. **Technical Implementation**

#### **Data Analysis & Preprocessing**
- Loaded and analyzed 232,725 music tracks from Spotify
- Created emotion labels using arousal-valence framework
- Implemented feature scaling and sequence generation
- Built comprehensive data visualization tools

#### **Model Architecture**
- **LSTM**: Input → Bi-LSTM(64) → Dropout → Bi-LSTM(32) → Dense(128) → Dense(64) → Softmax(4)
- **CNN**: Conv1D layers with MaxPooling and GlobalAveragePooling
- **Training**: Adam optimizer, early stopping, class balancing

#### **Evaluation Metrics**
- Accuracy, F1-macro, confusion matrices
- Cross-validation and model comparison
- Feature importance analysis
- Training history visualization

## 🚀 How to Use the System

### **Quick Start**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the system
python test_system.py

# 3. Train models
python train_model.py

# 4. Run the web app
streamlit run src/frontend/app.py

# 5. See a demo
python demo.py
```

### **Training Options**
```bash
# Basic training
python train_model.py

# Custom parameters
python train_model.py --epochs 100 --batch-size 16 --sequence-length 15

# Skip data analysis if already processed
python train_model.py --skip-analysis
```

## 📈 Expected Performance

Based on the architecture and methodology:

| Model | Expected Accuracy | Use Case |
|-------|------------------|----------|
| LSTM | 80-85% | Best for temporal patterns |
| 1D CNN | 75-80% | Good for sequence features |
| Random Forest | 70-75% | Fast baseline |
| SVM | 65-70% | Linear patterns |
| Logistic Regression | 60-65% | Simple baseline |

## 🎯 Key Achievements

### ✅ **Completed Requirements**
1. **EDA**: Comprehensive analysis of label balance and feature distributions
2. **LSTM Training**: Bidirectional LSTM for emotion classification
3. **Baseline Comparison**: 5 different baseline models implemented
4. **Streamlit Demo**: Interactive web interface for mood selection
5. **Documentation**: Detailed README and code documentation
6. **Clinical Considerations**: Ethical and clinical guidelines included

### 🔬 **Technical Highlights**
- **Temporal Modeling**: LSTM captures sequential patterns in audio features
- **Arousal-Valence Framework**: Psychologically grounded emotion classification
- **Real-time Prediction**: Live emotion analysis from audio features
- **Model Comparison**: Comprehensive benchmarking across multiple algorithms
- **User Experience**: Intuitive interface with visual feedback

## 🏥 Clinical & Ethical Considerations

### **Clinical Applications**
- Mood regulation and stress reduction
- Personalized music therapy interventions
- Mental health monitoring and tracking
- Therapeutic music recommendation

### **Ethical Guidelines**
- Privacy protection for emotional data
- Bias awareness and mitigation
- User consent and transparency
- Professional mental health integration
- Clear limitations and disclaimers

## 🛠️ System Requirements

### **Dependencies**
- Python 3.8+
- TensorFlow 2.15.0
- Streamlit 1.28.1
- scikit-learn 1.3.2
- pandas, numpy, matplotlib, seaborn, plotly

### **Hardware**
- RAM: 8GB+ recommended
- Storage: 2GB for models and data
- GPU: Optional but recommended for training

## 📝 Next Steps for Users

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Test System**: `python test_system.py`
3. **Train Models**: `python train_model.py`
4. **Run App**: `streamlit run src/frontend/app.py`
5. **Explore Demo**: `python demo.py`

## 🎉 Project Success

This project successfully demonstrates:

- **Advanced ML Implementation**: LSTM-based emotion classification
- **Real-world Application**: Practical music recommendation system
- **User-friendly Interface**: Modern Streamlit web application
- **Comprehensive Evaluation**: Multiple models and metrics
- **Clinical Relevance**: Mental health and therapeutic applications
- **Professional Documentation**: Complete setup and usage guides

The system is ready for immediate use and can be extended with additional features like real-time audio processing, user preference learning, and integration with music streaming APIs.

---

**Built with ❤️ for better mental health through music**

*Project completed: October 2024*
