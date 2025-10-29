"""
Main training script for the Emotion-Aware Music Recommendation System
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.utils.data_analysis import load_and_analyze_data, EmotionLabeler, visualize_emotion_distribution
from src.models.training_pipeline import TrainingPipeline

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train emotion classification models')
    parser.add_argument('--data-path', type=str, default='data/raw/SpotifyFeatures.csv',
                       help='Path to the dataset')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='Length of sequences for LSTM')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip data analysis and use existing processed data')
    
    args = parser.parse_args()
    
    print("="*60)
    print("EMOTION-AWARE MUSIC RECOMMENDATION SYSTEM")
    print("Training Pipeline")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    print(f"Data path: {args.data_path}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*60)
    
    # Step 1: Data Analysis and Preprocessing
    if not args.skip_analysis:
        print("\n1. Loading and analyzing data...")
        df = load_and_analyze_data(args.data_path)
        
        print("\n2. Creating emotion labels...")
        labeler = EmotionLabeler()
        df = labeler.create_emotion_labels(df)
        
        print("\n3. Analyzing emotion distribution...")
        emotion_counts = labeler.analyze_emotion_distribution(df)
        
        print("\n4. Creating visualizations...")
        visualize_emotion_distribution(df)
        
        print("\n5. Saving processed data...")
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv('data/processed/spotify_features_with_emotions.csv', index=False)
        print("Processed data saved to 'data/processed/spotify_features_with_emotions.csv'")
    
    # Step 2: Model Training
    print("\n6. Starting model training...")
    pipeline = TrainingPipeline('data/processed/spotify_features_with_emotions.csv')
    
    # Run complete training pipeline
    results = pipeline.run_complete_training(
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"End time: {datetime.now()}")
    
    # Display final results
    print("\nFinal Results Summary:")
    for model_name, result in results.items():
        if 'test_accuracy' in result:
            print(f"{model_name.upper()}: {result['test_accuracy']:.4f} accuracy")
    
    print("\nModels saved in 'models/' directory")
    print("You can now run the Streamlit app with: streamlit run src/frontend/app.py")

if __name__ == "__main__":
    main()
