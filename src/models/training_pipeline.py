"""
Training Pipeline for Emotion Classification
Orchestrates the training of LSTM and baseline models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
from datetime import datetime

from .lstm_model import EmotionLSTM, EmotionCNN1D
from .baseline_models import BaselineModels
from ..utils.data_analysis import DataPreprocessor

class TrainingPipeline:
    """
    Complete training pipeline for emotion classification
    """
    
    def __init__(self, data_path='data/processed/spotify_features_with_emotions.csv'):
        self.data_path = data_path
        self.preprocessor = DataPreprocessor()
        self.lstm_model = None
        self.cnn_model = None
        self.baseline_models = BaselineModels()
        self.results = {}
        
    def load_and_prepare_data(self, sequence_length=10, test_size=0.2, val_size=0.2):
        """Load and prepare data for training"""
        print("Loading and preparing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} samples")
        
        # Prepare features
        X_sequences, y_sequences = self.preprocessor.prepare_features(df, sequence_length)
        print(f"Created {len(X_sequences)} sequences of length {sequence_length}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_sequences, y_sequences, test_size=test_size, random_state=42, stratify=y_sequences
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Create balanced datasets
        X_train_balanced, y_train_balanced = self.preprocessor.create_balanced_dataset(
            X_train, y_train, max_samples_per_class=2000
        )
        
        print(f"Balanced train set: {len(X_train_balanced)} samples")
        
        return {
            'X_train': X_train_balanced,
            'y_train': y_train_balanced,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'sequence_length': sequence_length
        }
    
    def train_lstm(self, data, epochs=50, batch_size=32):
        """Train LSTM model"""
        print("\n" + "="*50)
        print("TRAINING LSTM MODEL")
        print("="*50)
        
        # Initialize LSTM model
        input_shape = (data['sequence_length'], data['X_train'].shape[2])
        self.lstm_model = EmotionLSTM(input_shape=input_shape)
        
        # Build and train model
        self.lstm_model.build_model()
        print(f"LSTM Model Summary:")
        self.lstm_model.model.summary()
        
        # Train model
        history = self.lstm_model.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            epochs=epochs, batch_size=batch_size
        )
        
        # Evaluate model
        lstm_results = self.lstm_model.evaluate(data['X_test'], data['y_test'])
        
        self.results['lstm'] = {
            'test_accuracy': lstm_results['test_accuracy'],
            'test_loss': lstm_results['test_loss'],
            'predictions': lstm_results['predictions'],
            'probabilities': lstm_results['probabilities']
        }
        
        print(f"\nLSTM Results:")
        print(f"Test Accuracy: {lstm_results['test_accuracy']:.4f}")
        print(f"Test Loss: {lstm_results['test_loss']:.4f}")
        
        # Plot training history
        self.lstm_model.plot_training_history('models/lstm_training_history.png')
        
        # Plot confusion matrix
        self.lstm_model.plot_confusion_matrix(
            data['y_test'], lstm_results['predictions'],
            'models/lstm_confusion_matrix.png'
        )
        
        return lstm_results
    
    def train_cnn1d(self, data, epochs=50, batch_size=32):
        """Train 1D CNN model"""
        print("\n" + "="*50)
        print("TRAINING 1D CNN MODEL")
        print("="*50)
        
        # Initialize CNN model
        input_shape = (data['sequence_length'], data['X_train'].shape[2])
        self.cnn_model = EmotionCNN1D(input_shape=input_shape)
        
        # Build and train model
        self.cnn_model.build_model()
        print(f"CNN Model Summary:")
        self.cnn_model.model.summary()
        
        # Train model
        history = self.cnn_model.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            epochs=epochs, batch_size=batch_size
        )
        
        # Evaluate model
        cnn_results = self.cnn_model.evaluate(data['X_test'], data['y_test'])
        
        self.results['cnn1d'] = {
            'test_accuracy': cnn_results['test_accuracy'],
            'test_loss': cnn_results['test_loss'],
            'predictions': cnn_results['predictions'],
            'probabilities': cnn_results['probabilities']
        }
        
        print(f"\nCNN Results:")
        print(f"Test Accuracy: {cnn_results['test_accuracy']:.4f}")
        print(f"Test Loss: {cnn_results['test_loss']:.4f}")
        
        return cnn_results
    
    def train_baselines(self, data):
        """Train baseline models"""
        print("\n" + "="*50)
        print("TRAINING BASELINE MODELS")
        print("="*50)
        
        # Train all baseline models
        baseline_results = self.baseline_models.train_all_baselines(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            data['X_test'], data['y_test']
        )
        
        # Compare models
        comparison_df = self.baseline_models.compare_models(data['y_test'])
        
        # Plot comparison
        self.baseline_models.plot_model_comparison('models/baseline_comparison.png')
        
        # Plot feature importance
        self.baseline_models.plot_feature_importance(save_path='models/feature_importance.png')
        
        self.results.update(baseline_results)
        return baseline_results
    
    def run_complete_training(self, sequence_length=10, epochs=50, batch_size=32):
        """Run complete training pipeline"""
        print("Starting complete training pipeline...")
        print(f"Timestamp: {datetime.now()}")
        
        # Load and prepare data
        data = self.load_and_prepare_data(sequence_length=sequence_length)
        
        # Train LSTM
        lstm_results = self.train_lstm(data, epochs=epochs, batch_size=batch_size)
        
        # Train CNN
        cnn_results = self.train_cnn1d(data, epochs=epochs, batch_size=batch_size)
        
        # Train baselines
        baseline_results = self.train_baselines(data)
        
        # Save models
        self.save_models()
        
        # Generate final report
        self.generate_final_report(data['y_test'])
        
        return self.results
    
    def save_models(self):
        """Save all trained models"""
        print("\nSaving models...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save LSTM model
        if self.lstm_model:
            self.lstm_model.save_model('models/lstm_emotion_model.h5')
        
        # Save CNN model
        if self.cnn_model:
            self.cnn_model.save_model('models/cnn1d_emotion_model.h5')
        
        # Save baseline models
        self.baseline_models.save_models('models/baseline_models')
        
        print("All models saved successfully!")
    
    def generate_final_report(self, y_test):
        """Generate final training report"""
        print("\n" + "="*60)
        print("FINAL TRAINING REPORT")
        print("="*60)
        
        # Create results summary
        results_summary = []
        
        for model_name, results in self.results.items():
            if 'test_accuracy' in results:
                results_summary.append({
                    'Model': model_name.upper(),
                    'Test Accuracy': f"{results['test_accuracy']:.4f}",
                    'Test Loss': f"{results.get('test_loss', 'N/A')}"
                })
        
        results_df = pd.DataFrame(results_summary)
        print("\nModel Performance Summary:")
        print(results_df.to_string(index=False))
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1].get('test_accuracy', 0))
        print(f"\nBest Model: {best_model[0].upper()} with accuracy {best_model[1]['test_accuracy']:.4f}")
        
        # Save results to JSON
        results_json = {}
        for model_name, results in self.results.items():
            results_json[model_name] = {
                'test_accuracy': float(results['test_accuracy']),
                'test_loss': float(results.get('test_loss', 0))
            }
        
        with open('models/training_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\nResults saved to 'models/training_results.json'")
        
        return results_df
    
    def plot_all_confusion_matrices(self, y_test, save_path='models/all_confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        n_models = len([r for r in self.results.values() if 'predictions' in r])
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        model_idx = 0
        for model_name, results in self.results.items():
            if 'predictions' in results:
                cm = confusion_matrix(y_test, results['predictions'])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[model_idx])
                axes[model_idx].set_title(f'{model_name.upper()}\nAccuracy: {results["test_accuracy"]:.3f}')
                axes[model_idx].set_xlabel('Predicted')
                axes[model_idx].set_ylabel('Actual')
                
                model_idx += 1
        
        # Hide unused subplots
        for i in range(model_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Run training pipeline
    pipeline = TrainingPipeline()
    results = pipeline.run_complete_training(sequence_length=10, epochs=30, batch_size=32)
