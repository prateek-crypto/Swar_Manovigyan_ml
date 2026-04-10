#!/usr/bin/env python
"""
Retrain baseline models with current scikit-learn version to fix compatibility warnings.
Resolves InconsistentVersionWarning for sklearn models trained with 1.6.1, loaded with 1.7.2+
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.training_pipeline import TrainingPipeline

def main():
    print("Retraining baseline models with current scikit-learn version...")
    print("This will update the baseline models to be compatible with sklearn 1.7.2+\n")
    
    # Create and run training pipeline
    pipeline = TrainingPipeline(data_path='data/processed/spotify_features_with_emotions.csv')
    
    # Load and prepare data
    print("Loading training data...")
    data = pipeline.load_and_prepare_data(sequence_length=10)
    
    # Train baseline models only
    print("\nTraining baseline models...")
    baseline_results = pipeline.train_baselines(data)
    
    # Save baseline models to disk
    print("\nSaving baseline models...")
    pipeline.baseline_models.save_models('models/baseline_models')
    
    print("\n✓ Baseline models retrained and saved successfully!")
    print("\nResults:")
    for model_name, results in baseline_results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Train Accuracy: {results['train_accuracy']:.4f}")
        print(f"  Val Accuracy:   {results['val_accuracy']:.4f}")
        print(f"  Test Accuracy:  {results['test_accuracy']:.4f}")

if __name__ == '__main__':
    main()
