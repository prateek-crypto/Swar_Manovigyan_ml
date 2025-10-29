"""
Baseline training script for the Emotion-Aware Music Recommendation System
Trains only baseline models without requiring TensorFlow
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.utils.data_analysis import load_and_analyze_data, EmotionLabeler, DataPreprocessor
from src.models.baseline_models import BaselineModels

def main():
    """Main training function for baseline models only"""
    parser = argparse.ArgumentParser(description='Train baseline emotion classification models')
    parser.add_argument('--data-path', type=str, default='data/raw/SpotifyFeatures.csv',
                       help='Path to the dataset')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='Validation set size')
    parser.add_argument('--max-samples', type=int, default=5000,
                       help='Maximum samples per class')
    
    args = parser.parse_args()
    
    print("="*60)
    print("EMOTION-AWARE MUSIC RECOMMENDATION SYSTEM")
    print("BASELINE MODELS TRAINING")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    print(f"Data path: {args.data_path}")
    print("="*60)
    
    # Step 1: Load and prepare data
    print("\n1. Loading and preparing data...")
    try:
        df = load_and_analyze_data(args.data_path)
        print(f"Loaded {len(df)} samples")
        
        # Create emotion labels
        labeler = EmotionLabeler()
        df = labeler.create_emotion_labels(df)
        print("Created emotion labels")
        
        # Analyze emotion distribution
        emotion_counts = labeler.analyze_emotion_distribution(df)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Step 2: Prepare features
    print("\n2. Preparing features...")
    try:
        preprocessor = DataPreprocessor()
        X_sequences, y_sequences = preprocessor.prepare_features(df, sequence_length=10)
        print(f"Created {len(X_sequences)} sequences of shape {X_sequences.shape}")
        
        # Create balanced dataset
        X_balanced, y_balanced = preprocessor.create_balanced_dataset(
            X_sequences, y_sequences, max_samples_per_class=args.max_samples
        )
        print(f"Balanced dataset: {len(X_balanced)} samples")
        
    except Exception as e:
        print(f"Error preparing features: {e}")
        return False
    
    # Step 3: Split data
    print("\n3. Splitting data...")
    try:
        from sklearn.model_selection import train_test_split
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_balanced, y_balanced, test_size=args.test_size, random_state=42, stratify=y_balanced
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=args.val_size/(1-args.test_size), random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
    except Exception as e:
        print(f"Error splitting data: {e}")
        return False
    
    # Step 4: Train baseline models
    print("\n4. Training baseline models...")
    try:
        baseline_models = BaselineModels()
        
        # Train all baseline models
        results = baseline_models.train_all_baselines(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Compare models
        comparison_df = baseline_models.compare_models(y_test)
        
        # Save models
        baseline_models.save_models('models/baseline_models')
        
    except Exception as e:
        print(f"Error training models: {e}")
        return False
    
    # Step 5: Generate report
    print("\n5. Generating training report...")
    try:
        print("\n" + "="*60)
        print("TRAINING RESULTS")
        print("="*60)
        
        for model_name, result in results.items():
            if 'test_accuracy' in result:
                print(f"{model_name.upper()}: {result['test_accuracy']:.4f} accuracy")
        
        # Save results
        import json
        results_json = {}
        for model_name, result in results.items():
            results_json[model_name] = {
                'test_accuracy': float(result['test_accuracy']),
                'val_accuracy': float(result['val_accuracy']),
                'train_accuracy': float(result['train_accuracy'])
            }
        
        os.makedirs('models', exist_ok=True)
        with open('models/baseline_training_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\nResults saved to 'models/baseline_training_results.json'")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return False
    
    print("\n" + "="*60)
    print("BASELINE TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"End time: {datetime.now()}")
    print("\nNext steps:")
    print("1. Run 'streamlit run src/frontend/app.py' to start the app")
    print("2. Run 'python demo.py' to see a demonstration")
    print("3. Note: LSTM models require TensorFlow (optional)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
