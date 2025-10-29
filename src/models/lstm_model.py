"""
LSTM-based Emotion Classification Model
Implements bidirectional LSTM for emotion classification using audio features
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EmotionLSTM:
    """
    LSTM-based emotion classification model
    """
    
    def __init__(self, input_shape, num_classes=4, lstm_units=64, dropout_rate=0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the LSTM model architecture"""
        model = Sequential([
            # Input layer
            tf.keras.layers.Input(shape=self.input_shape),
            
            # First LSTM layer with dropout
            Bidirectional(
                LSTM(self.lstm_units, return_sequences=True),
                name='bidirectional_lstm_1'
            ),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            
            # Second LSTM layer
            Bidirectional(
                LSTM(self.lstm_units // 2, return_sequences=False),
                name='bidirectional_lstm_2'
            ),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            
            # Dense layers
            Dense(128, activation='relu', name='dense_1'),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            
            Dense(64, activation='relu', name='dense_2'),
            Dropout(self.dropout_rate),
            
            # Output layer
            Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1):
        """Train the LSTM model"""
        if self.model is None:
            self.build_model()
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_val_cat = to_categorical(y_val, self.num_classes)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Convert test labels to categorical for loss calculation
        y_test_cat = to_categorical(y_test, self.num_classes)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_cat, verbose=0)
        
        # Classification report
        class_names = ['Low Arousal, Negative Valence', 'Low Arousal, Positive Valence',
                      'High Arousal, Negative Valence', 'High Arousal, Positive Valence']
        
        report = classification_report(y_test, y_pred, target_names=class_names)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': report
        }
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low Arousal, Negative Valence', 'Low Arousal, Positive Valence',
                               'High Arousal, Negative Valence', 'High Arousal, Positive Valence'],
                   yticklabels=['Low Arousal, Negative Valence', 'Low Arousal, Positive Valence',
                               'High Arousal, Negative Valence', 'High Arousal, Positive Valence'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict_emotion(self, X):
        """Predict emotion for given input"""
        if self.model is None:
            raise ValueError("Model not loaded. Load a trained model first.")
        
        predictions = self.model.predict(X)
        emotion_labels = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        return emotion_labels, confidence_scores, predictions

class EmotionCNN1D:
    """
    1D CNN baseline model for emotion classification
    """
    
    def __init__(self, input_shape, num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build 1D CNN model"""
        model = Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            
            # First Conv1D block
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.3),
            
            # Second Conv1D block
            tf.keras.layers.Conv1D(128, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.3),
            
            # Third Conv1D block
            tf.keras.layers.Conv1D(256, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.3),
            
            # Dense layers
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
