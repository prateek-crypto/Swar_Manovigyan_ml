"""
Data Analysis and Emotion Labeling Module
Creates emotion labels based on arousal-valence framework using Spotify features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class EmotionLabeler:
    """
    Creates emotion labels based on arousal-valence framework
    using Spotify audio features
    """
    
    def __init__(self):
        self.emotion_mapping = {
            0: 'Low Arousal, Negative Valence',  # Sad, Depressed
            1: 'Low Arousal, Positive Valence',  # Calm, Peaceful
            2: 'High Arousal, Negative Valence', # Angry, Stressed
            3: 'High Arousal, Positive Valence'  # Happy, Excited
        }
        
    def create_emotion_labels(self, df):
        """
        Create emotion labels based on arousal-valence framework
        
        Arousal: Based on energy, loudness, tempo
        Valence: Based on valence, danceability, acousticness
        """
        # Calculate arousal score (0-1)
        arousal_features = ['energy', 'loudness', 'tempo']
        arousal_data = df[arousal_features].copy()
        
        # Normalize loudness (typically negative values)
        arousal_data['loudness'] = (arousal_data['loudness'] - arousal_data['loudness'].min()) / \
                                  (arousal_data['loudness'].max() - arousal_data['loudness'].min())
        
        # Normalize tempo
        arousal_data['tempo'] = (arousal_data['tempo'] - arousal_data['tempo'].min()) / \
                               (arousal_data['tempo'].max() - arousal_data['tempo'].min())
        
        # Calculate arousal as weighted average
        arousal_weights = [0.4, 0.3, 0.3]  # energy, loudness, tempo
        df['arousal'] = np.average(arousal_data, axis=1, weights=arousal_weights)
        
        # Calculate valence score (0-1) - Spotify already provides valence
        # But we can enhance it with other features
        valence_features = ['valence', 'danceability', 'acousticness']
        valence_data = df[valence_features].copy()
        
        # Invert acousticness for valence (more acoustic = less energetic/positive)
        valence_data['acousticness'] = 1 - valence_data['acousticness']
        
        # Calculate enhanced valence
        valence_weights = [0.6, 0.3, 0.1]  # valence, danceability, acousticness
        df['enhanced_valence'] = np.average(valence_data, axis=1, weights=valence_weights)
        
        # Create binary arousal and valence labels
        df['arousal_binary'] = (df['arousal'] > df['arousal'].median()).astype(int)
        df['valence_binary'] = (df['enhanced_valence'] > df['enhanced_valence'].median()).astype(int)
        
        # Create 4-class emotion labels
        df['emotion_label'] = df['arousal_binary'] * 2 + df['valence_binary']
        
        return df
    
    def analyze_emotion_distribution(self, df):
        """Analyze the distribution of emotion labels"""
        emotion_counts = df['emotion_label'].value_counts().sort_index()
        
        print("Emotion Label Distribution:")
        for label, count in emotion_counts.items():
            emotion_name = self.emotion_mapping[label]
            percentage = (count / len(df)) * 100
            print(f"{label}: {emotion_name} - {count} samples ({percentage:.2f}%)")
        
        return emotion_counts

class DataPreprocessor:
    """
    Preprocesses audio features for LSTM training
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def prepare_features(self, df, sequence_length=10):
        """
        Prepare features for LSTM training
        
        Args:
            df: DataFrame with audio features
            sequence_length: Length of sequences for LSTM
        """
        # Select relevant audio features
        feature_columns = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
            'arousal', 'enhanced_valence'
        ]
        
        # Handle missing values
        df_features = df[feature_columns].fillna(df[feature_columns].median())
        
        # Scale features
        scaled_features = self.scaler.fit_transform(df_features)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(scaled_features) - sequence_length + 1):
            X_sequences.append(scaled_features[i:i+sequence_length])
            y_sequences.append(df['emotion_label'].iloc[i+sequence_length-1])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def create_balanced_dataset(self, X, y, max_samples_per_class=5000):
        """
        Create a balanced dataset by sampling equal number of samples per class
        """
        unique_classes = np.unique(y)
        balanced_X = []
        balanced_y = []
        
        for class_label in unique_classes:
            class_indices = np.where(y == class_label)[0]
            n_samples = min(len(class_indices), max_samples_per_class)
            selected_indices = np.random.choice(class_indices, n_samples, replace=False)
            
            balanced_X.append(X[selected_indices])
            balanced_y.append(y[selected_indices])
        
        return np.vstack(balanced_X), np.hstack(balanced_y)

def load_and_analyze_data(file_path):
    """Load and perform initial analysis of the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

def visualize_emotion_distribution(df):
    """Create visualizations for emotion distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Arousal vs Valence scatter plot
    scatter = axes[0, 0].scatter(df['arousal'], df['enhanced_valence'], 
                                c=df['emotion_label'], cmap='viridis', alpha=0.6)
    axes[0, 0].set_xlabel('Arousal')
    axes[0, 0].set_ylabel('Valence')
    axes[0, 0].set_title('Arousal vs Valence Distribution')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # Emotion label distribution
    emotion_counts = df['emotion_label'].value_counts().sort_index()
    axes[0, 1].bar(emotion_counts.index, emotion_counts.values)
    axes[0, 1].set_xlabel('Emotion Label')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Emotion Label Distribution')
    axes[0, 1].set_xticks(range(4))
    
    # Feature correlation heatmap
    feature_cols = ['acousticness', 'danceability', 'energy', 'valence', 
                   'arousal', 'enhanced_valence']
    corr_matrix = df[feature_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 0])
    axes[1, 0].set_title('Feature Correlation Matrix')
    
    # Genre distribution
    genre_counts = df['genre'].value_counts().head(10)
    axes[1, 1].barh(range(len(genre_counts)), genre_counts.values)
    axes[1, 1].set_yticks(range(len(genre_counts)))
    axes[1, 1].set_yticklabels(genre_counts.index)
    axes[1, 1].set_xlabel('Count')
    axes[1, 1].set_title('Top 10 Genres')
    
    plt.tight_layout()
    plt.savefig('data/processed/emotion_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load and analyze data
    df = load_and_analyze_data('data/raw/SpotifyFeatures.csv')
    
    # Create emotion labels
    labeler = EmotionLabeler()
    df = labeler.create_emotion_labels(df)
    
    # Analyze emotion distribution
    emotion_counts = labeler.analyze_emotion_distribution(df)
    
    # Create visualizations
    visualize_emotion_distribution(df)
    
    # Save processed data
    df.to_csv('data/processed/spotify_features_with_emotions.csv', index=False)
    print(f"\nProcessed data saved to 'data/processed/spotify_features_with_emotions.csv'")
    print(f"Total samples: {len(df)}")
    print(f"Features: {df.shape[1]}")
