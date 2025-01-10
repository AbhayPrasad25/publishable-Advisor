from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime
import joblib

class ModelEvaluator:
    def __init__(self, 
                 model_dir: str,
                 results_dir: str,
                 log_path: Optional[str] = None):
        """
        Initialize the model evaluator
        
        Args:
            model_dir: Directory containing trained model
            results_dir: Directory to save evaluation results
            log_path: Path to log file (optional)
        """
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging(log_path)
        
        # Load model and scaler
        self.model = joblib.load(self.model_dir / 'model.joblib')
        self.scaler = joblib.load(self.model_dir / 'scaler.joblib')

    def _setup_logging(self, log_path: Optional[str]) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ModelEvaluator')
        logger.setLevel(logging.INFO)
        
        if log_path:
            handler = logging.FileHandler(log_path)
        else:
            handler = logging.StreamHandler()
            
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def evaluate_predictions(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           y_prob: np.ndarray) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_prob[:, 1]),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        return metrics

    def plot_confusion_matrix(self, 
                            confusion_mat: np.ndarray,
                            output_path: str) -> None:
        """
        Plot confusion matrix heatmap
        
        Args:
            confusion_mat: Confusion matrix array
            output_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mat, 
                   annot=True, 
                   fmt='d',
                   cmap='Blues',
                   xticklabels=['Non-publishable', 'Publishable'],
                   yticklabels=['Non-publishable', 'Publishable'])
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_feature_importance(self, 
                              feature_names: List[str],
                              importance_scores: np.ndarray,
                              output_path: str,
                              top_n: int = 20) -> None:
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            importance_scores: Feature importance scores
            output_path: Path to save the plot
            top_n: Number of top features to show
        """
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Most Important Features')
        plt.barh(range(top_n), importance_scores[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def analyze_errors(self, 
                      X_test: np.ndarray,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      feature_names: List[str]) -> Dict:
        """
        Analyze prediction errors
        
        Args:
            X_test: Test features
            y_true: True labels
            y_pred: Predicted labels
            feature_names: List of feature names
            
        Returns:
            Dictionary containing error analysis
        """
        # Find incorrect predictions
        error_indices = np.where(y_true != y_pred)[0]
        
        error_analysis = {
            'num_errors': len(error_indices),
            'error_rate': len(error_indices) / len(y_true),
            'error_details': []
        }
        
        # Analyze each error
        for idx in error_indices:
            error_details = {
                'index': int(idx),
                'true_label': int(y_true[idx]),
                'predicted_label': int(y_pred[idx]),
                'feature_values': dict(zip(feature_names, X_test[idx].tolist()))
            }
            error_analysis['error_details'].append(error_details)
        
        return error_analysis

    def evaluate_model(self, 
                      test_features: str,
                      test_labels: str) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            test_features: Path to test features JSON
            test_labels: Path to test labels JSON
            
        Returns:
            Dictionary containing all evaluation results
        """
        # Load test data
        with open(test_features, 'r') as f:
            features_list = json.load(f)
        
        with open(test_labels, 'r') as f:
            y_true = np.array(json.load(f))
        
        # Prepare features
        X_test = np.array([list(f.values()) for f in features_list])
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get predictions
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        metrics = self.evaluate_predictions(y_true, y_pred, y_prob)
        
        # Generate plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Confusion matrix plot
        conf_matrix_path = self.results_dir / f'confusion_matrix_{timestamp}.png'
        self.plot_confusion_matrix(
            np.array(metrics['confusion_matrix']),
            str(conf_matrix_path)
        )
        
        # Feature importance plot
        if hasattr(self.model, 'feature_importances_'):
            feature_names = list(features_list[0].keys())
            importance_path = self.results_dir / f'feature_importance_{timestamp}.png'
            self.plot_feature_importance(
                feature_names,
                self.model.feature_importances_,
                str(importance_path)
            )
        
        # Error analysis
        error_analysis = self.analyze_errors(
            X_test_scaled,
            y_true,
            y_pred,
            list(features_list[0].keys())
        )
        
        # Compile full evaluation report
        evaluation_report = {
            'metrics': metrics,
            'error_analysis': error_analysis,
            'plots': {
                'confusion_matrix': str(conf_matrix_path),
                'feature_importance': str(importance_path) if hasattr(self.model, 'feature_importances_') else None
            }
        }
        
        # Save report
        report_path = self.results_dir / f'evaluation_report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        self.logger.info(f"Evaluation completed. Report saved to {report_path}")
        return evaluation_report

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory containing trained model')
    parser.add_argument('--test_features', type=str, required=True,
                      help='Path to test features JSON')
    parser.add_argument('--test_labels', type=str, required=True,
                      help='Path to test labels JSON')
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory to save evaluation results')
    parser.add_argument('--log_path', type=str,
                      help='Path to log file (optional)')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        log_path=args.log_path
    )
    
    evaluator.evaluate_model(args.test_features, args.test_labels)

if __name__ == "__main__":
    main()