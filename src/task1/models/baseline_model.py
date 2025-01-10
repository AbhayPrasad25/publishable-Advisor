from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import json
import logging
from pathlib import Path

class BaselineModel:
    def __init__(self, 
                 model_path: Optional[str] = None,
                 random_state: int = 42,
                 log_path: Optional[str] = None):
        """
        Initialize the baseline model for paper classification
        
        Args:
            model_path: Path to load pre-trained model (optional)
            random_state: Random seed for reproducibility
            log_path: Path to log file (optional)
        """
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.logger = self._setup_logging(log_path)
        
        if model_path:
            self.load_model(model_path)

    def _setup_logging(self, log_path: Optional[str]) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('BaselineModel')
        logger.setLevel(logging.INFO)
        
        if log_path:
            handler = logging.FileHandler(log_path)
        else:
            handler = logging.StreamHandler()
            
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def prepare_features(self, features_list: List[Dict]) -> np.ndarray:
        """
        Convert feature dictionaries to numpy array
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            numpy array of features
        """
        # Extract numeric features and flatten embeddings
        processed_features = []
        
        for features in features_list:
            feature_vector = []
            
            # Process each feature
            for key, value in features.items():
                # Skip embedding features for now - they'll be handled separately
                if key in ['tfidf_embedding', 'doc2vec_embedding']:
                    continue
                    
                # Convert numeric features
                if isinstance(value, (int, float)):
                    feature_vector.append(value)
            
            # Add embedding features if they exist
            if 'tfidf_embedding' in features:
                feature_vector.extend(features['tfidf_embedding'])
            if 'doc2vec_embedding' in features:
                feature_vector.extend(features['doc2vec_embedding'])
            
            processed_features.append(feature_vector)
        
        return np.array(processed_features)

    def train(self, 
              features_path: str,
              labels_path: str,
              test_size: float = 0.2) -> Dict:
        """
        Train the model using extracted features
        
        Args:
            features_path: Path to features JSON file
            labels_path: Path to labels JSON file
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing training metrics
        """
        # Load features and labels
        with open(features_path, 'r') as f:
            features_list = json.load(f)
        
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        # Prepare feature matrix
        X = self.prepare_features(features_list)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        # Detailed test set evaluation
        y_pred = self.model.predict(X_test_scaled)
        classification_metrics = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        # Feature importance analysis
        feature_importance = self.model.feature_importances_.tolist()
        
        # Compile metrics
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_metrics': classification_metrics,
            'confusion_matrix': conf_matrix,
            'feature_importance': feature_importance
        }
        
        self.logger.info(f"Training completed. Test accuracy: {test_score:.4f}")
        return metrics

    def predict(self, features: Dict) -> Tuple[int, float]:
        """
        Make prediction for a single paper
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Prepare features
        X = self.prepare_features([features])
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probability
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = probabilities[prediction]
        
        return prediction, confidence

    def batch_predict(self, features_path: str, output_path: str) -> None:
        """
        Make predictions for multiple papers
        
        Args:
            features_path: Path to features JSON file
            output_path: Path to save predictions
        """
        # Load features
        with open(features_path, 'r') as f:
            features_list = json.load(f)
        
        # Prepare features
        X = self.prepare_features(features_list)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Compile results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            results.append({
                'paper_id': i,
                'prediction': int(pred),
                'confidence': float(probs[pred]),
                'probabilities': probs.tolist()
            })
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Processed {len(results)} predictions")

    def save_model(self, model_dir: str) -> None:
        """Save model and scaler"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, model_dir / 'model.joblib')
        joblib.dump(self.scaler, model_dir / 'scaler.joblib')
        
        self.logger.info(f"Model saved to {model_dir}")

    def load_model(self, model_dir: str) -> None:
        """Load model and scaler"""
        model_dir = Path(model_dir)
        
        self.model = joblib.load(model_dir / 'model.joblib')
        self.scaler = joblib.load(model_dir / 'scaler.joblib')
        
        self.logger.info(f"Model loaded from {model_dir}")