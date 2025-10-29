"""
Baseline Models for Emotion Classification
Implements Logistic Regression, Random Forest, and other baseline models
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class BaselineModels:
    """
    Collection of baseline models for emotion classification
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def prepare_data_for_baseline(self, X_sequences, y_sequences):
        """
        Prepare sequence data for baseline models by flattening sequences
        """
        # Flatten sequences for traditional ML models
        X_flattened = X_sequences.reshape(X_sequences.shape[0], -1)
        return X_flattened, y_sequences
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train Logistic Regression model"""
        print("Training Logistic Regression...")
        
        # Flatten data
        X_train_flat, _ = self.prepare_data_for_baseline(X_train, y_train)
        X_val_flat, _ = self.prepare_data_for_baseline(X_val, y_val)
        X_test_flat, _ = self.prepare_data_for_baseline(X_test, y_test)
        
        # Train model
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        lr_model.fit(X_train_flat, y_train)
        
        # Evaluate
        train_pred = lr_model.predict(X_train_flat)
        val_pred = lr_model.predict(X_val_flat)
        test_pred = lr_model.predict(X_test_flat)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        self.models['logistic_regression'] = lr_model
        self.results['logistic_regression'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'predictions': test_pred
        }
        
        print(f"Logistic Regression - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        return lr_model
    
    def train_random_forest(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        # Flatten data
        X_train_flat, _ = self.prepare_data_for_baseline(X_train, y_train)
        X_val_flat, _ = self.prepare_data_for_baseline(X_val, y_val)
        X_test_flat, _ = self.prepare_data_for_baseline(X_test, y_test)
        
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        rf_model.fit(X_train_flat, y_train)
        
        # Evaluate
        train_pred = rf_model.predict(X_train_flat)
        val_pred = rf_model.predict(X_val_flat)
        test_pred = rf_model.predict(X_test_flat)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'predictions': test_pred
        }
        
        print(f"Random Forest - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        return rf_model
    
    def train_svm(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train Support Vector Machine model"""
        print("Training SVM...")
        
        # Flatten data
        X_train_flat, _ = self.prepare_data_for_baseline(X_train, y_train)
        X_val_flat, _ = self.prepare_data_for_baseline(X_val, y_val)
        X_test_flat, _ = self.prepare_data_for_baseline(X_test, y_test)
        
        # Train model
        svm_model = SVC(
            kernel='rbf',
            random_state=42,
            class_weight='balanced',
            probability=True
        )
        svm_model.fit(X_train_flat, y_train)
        
        # Evaluate
        train_pred = svm_model.predict(X_train_flat)
        val_pred = svm_model.predict(X_val_flat)
        test_pred = svm_model.predict(X_test_flat)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        self.models['svm'] = svm_model
        self.results['svm'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'predictions': test_pred
        }
        
        print(f"SVM - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        return svm_model
    
    def train_mlp(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train Multi-layer Perceptron model"""
        print("Training MLP...")
        
        # Flatten data
        X_train_flat, _ = self.prepare_data_for_baseline(X_train, y_train)
        X_val_flat, _ = self.prepare_data_for_baseline(X_val, y_val)
        X_test_flat, _ = self.prepare_data_for_baseline(X_test, y_test)
        
        # Train model
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            random_state=42,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1
        )
        mlp_model.fit(X_train_flat, y_train)
        
        # Evaluate
        train_pred = mlp_model.predict(X_train_flat)
        val_pred = mlp_model.predict(X_val_flat)
        test_pred = mlp_model.predict(X_test_flat)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        self.models['mlp'] = mlp_model
        self.results['mlp'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'predictions': test_pred
        }
        
        print(f"MLP - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        return mlp_model
    
    def train_all_baselines(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train all baseline models"""
        print("Training all baseline models...")
        
        # Train each model
        self.train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
        self.train_svm(X_train, y_train, X_val, y_val, X_test, y_test)
        self.train_mlp(X_train, y_train, X_val, y_val, X_test, y_test)
        
        return self.results
    
    def compare_models(self, y_test):
        """Compare performance of all models"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Test Accuracy': f"{results['test_accuracy']:.4f}",
                'Validation Accuracy': f"{results['val_accuracy']:.4f}",
                'Training Accuracy': f"{results['train_accuracy']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_model_comparison(self, save_path=None):
        """Plot model comparison"""
        if not self.results:
            print("No results to plot. Train models first.")
            return
        
        models = list(self.results.keys())
        test_accs = [self.results[model]['test_accuracy'] for model in models]
        val_accs = [self.results[model]['val_accuracy'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
        bars2 = ax.bar(x + width/2, val_accs, width, label='Validation Accuracy', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([model.replace('_', ' ').title() for model in models], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_models(self, directory='models/baseline_models'):
        """Save all trained models"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = os.path.join(directory, f'{model_name}.joblib')
            joblib.dump(model, filepath)
            print(f"Saved {model_name} to {filepath}")
    
    def load_models(self, directory='models/baseline_models'):
        """Load all trained models"""
        import os
        for model_name in ['logistic_regression', 'random_forest', 'svm', 'mlp']:
            filepath = os.path.join(directory, f'{model_name}.joblib')
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
                print(f"Loaded {model_name} from {filepath}")
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance from Random Forest"""
        if 'random_forest' not in self.models:
            print("Random Forest model not found. Train it first.")
            return None
        
        rf_model = self.models['random_forest']
        importances = rf_model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        return importance_df
    
    def plot_feature_importance(self, feature_names=None, top_n=20, save_path=None):
        """Plot feature importance"""
        importance_df = self.get_feature_importance(feature_names)
        
        if importance_df is None:
            return
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features (Random Forest)')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
