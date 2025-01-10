import argparse
from pathlib import Path
import json
import logging
from datetime import datetime

from src.task1.data.pdf_extractor import PdfExtractor
from src.task1.data.preprocessing import Preprocessor
from src.task1.data.feature_extraction import FeatureExtractor
from src.task1.models.baseline_model import BaselineModel

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logger = logging.getLogger('TrainingPipeline')
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def main():
    parser = argparse.ArgumentParser(description='Train paper classification model')
    parser.add_argument('--raw_dir', type=str, required=True,
                      help='Directory containing raw PDF files')
    parser.add_argument('--labeled_dir', type=str, required=True,
                      help='Directory containing labeled PDFs')
    parser.add_argument('--processed_dir', type=str, required=True,
                      help='Directory for processed data')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory for saving model files')
    parser.add_argument('--log_dir', type=str, required=True,
                      help='Directory for log files')
    
    args = parser.parse_args()
    logger = setup_logging(args.log_dir)
    
    try:
        # Step 1: Extract text from PDFs
        logger.info("Starting PDF text extraction...")
        extractor = PdfExtractor(log_path=str(Path(args.log_dir) / "extractor.log"))
        
        # Process labeled data
        labeled_output_dir = Path(args.processed_dir) / "labeled"
        labeled_output_dir.mkdir(parents=True, exist_ok=True)
        extractor.batch_process(args.labeled_dir, str(labeled_output_dir))
        
        # Step 2: Preprocess extracted text
        logger.info("Starting text preprocessing...")
        preprocessor = Preprocessor(log_path=str(Path(args.log_dir) / "preprocessor.log"))
        
        preprocessed_dir = Path(args.processed_dir) / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        preprocessor.batch_process(str(labeled_output_dir), str(preprocessed_dir))
        
        # Step 3: Extract features
        logger.info("Starting feature extraction...")
        feature_extractor = FeatureExtractor(
            use_tfidf=True,
            use_doc2vec=True,
            log_path=str(Path(args.log_dir) / "feature_extractor.log")
        )
        
        features_dir = Path(args.processed_dir) / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        feature_extractor.batch_process(
            str(preprocessed_dir),
            str(features_dir),
            extract_embeddings=True
        )
        
        # Step 4: Train model
        logger.info("Starting model training...")
        model = BaselineModel(
            log_path=str(Path(args.log_dir) / "model.log")
        )
        
        features_file = features_dir / "all_features.json"
        labels_file = Path(args.labeled_dir) / "labels.json"
        
        metrics = model.train(str(features_file), str(labels_file))
        
        # Save model and metrics
        model.save_model(args.model_dir)
        
        metrics_file = Path(args.model_dir) / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()