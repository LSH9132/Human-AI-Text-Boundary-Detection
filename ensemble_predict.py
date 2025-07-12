#!/usr/bin/env python3
"""
Ensemble prediction script for AI Text Detection.
Combines multiple trained models for improved performance.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict
import glob

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import get_config_for_model, Config
from src.data_processor import DataProcessor
from src.predictor import Predictor


def setup_logging(log_file: str = "ensemble_prediction.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def find_model_files(model_dir: str = "models") -> Dict[str, List[str]]:
    """Find trained model files grouped by model type."""
    model_files = {}
    
    # Look for different model patterns
    patterns = {
        'koelectra': 'best_model_fold_*.pt',  # KoELECTRA models in main directory
        'kcbert': '*kcbert*fold*.pt', 
        'klue': '*klue*fold*.pt',
        'bert': '*bert*fold*.pt'
    }
    
    for model_type, pattern in patterns.items():
        files = glob.glob(os.path.join(model_dir, pattern))
        if files:
            model_files[model_type] = sorted(files)
    
    return model_files


def main():
    parser = argparse.ArgumentParser(description='Ensemble prediction for AI text detection')
    parser.add_argument('--model-dir', default='models', help='Directory containing trained models')
    parser.add_argument('--models', nargs='+', help='Specific model types to use (koelectra, kcbert, klue)')
    parser.add_argument('--method', default='weighted_mean', choices=['mean', 'median', 'weighted_mean'],
                       help='Ensemble method')
    parser.add_argument('--optimize-weights', action='store_true', 
                       help='Optimize ensemble weights using validation data')
    parser.add_argument('--output', default='ensemble_submission.csv', help='Output submission file')
    parser.add_argument('--config-model', default='koelectra', help='Base model for configuration')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 ENSEMBLE PREDICTION FOR AI TEXT DETECTION")
    logger.info("=" * 60)
    
    # Load configuration
    config = get_config_for_model(args.config_model)
    
    # Find available models
    model_files = find_model_files(args.model_dir)
    
    if not model_files:
        logger.error("No trained models found in models directory")
        return
    
    logger.info(f"Found model groups: {list(model_files.keys())}")
    for model_type, files in model_files.items():
        logger.info(f"  {model_type}: {len(files)} models")
    
    # Select models to use
    if args.models:
        selected_models = {k: v for k, v in model_files.items() if k in args.models}
    else:
        selected_models = model_files
    
    if not selected_models:
        logger.error("No models selected for ensemble")
        return
    
    # Collect all model paths
    all_model_paths = []
    model_info = []
    
    for model_type, paths in selected_models.items():
        for path in paths:
            all_model_paths.append(path)
            model_info.append({
                'path': path,
                'type': model_type,
                'fold': os.path.basename(path)
            })
    
    logger.info(f"Using {len(all_model_paths)} models for ensemble:")
    for info in model_info:
        logger.info(f"  {info['type']}: {info['fold']}")
    
    # Load test data
    logger.info("📊 Loading test data...")
    data_processor = DataProcessor(config.data, config.model)
    _, test_df = data_processor.load_data()
    
    logger.info(f"Loaded {len(test_df)} test samples")
    
    # Initialize predictor with the first model's config
    first_model_type = list(selected_models.keys())[0]
    config = get_config_for_model(first_model_type)
    predictor = Predictor(config)
    
    # Optimize weights if requested
    weights = None
    if args.optimize_weights:
        logger.info("🔧 Optimizing ensemble weights...")
        # For now, use equal weights (could be improved with validation data)
        # In a real scenario, you'd need validation data with labels
        weights = [1.0] * len(all_model_paths)
        logger.info("Weight optimization requires validation data - using equal weights")
    
    # Perform ensemble prediction
    logger.info(f"🔮 Performing ensemble prediction using {args.method}...")
    
    try:
        predictions = predictor.ensemble_predict(
            all_model_paths, 
            test_df, 
            method=args.method,
            weights=weights
        )
        
        # Create submission
        submission_df = pd.DataFrame({
            'id': test_df.index,
            'generated': predictions
        })
        
        # Save submission
        submission_df.to_csv(args.output, index=False)
        logger.info(f"✅ Ensemble predictions saved to {args.output}")
        
        # Log prediction statistics
        logger.info(f"📈 Prediction statistics:")
        logger.info(f"  Mean: {np.mean(predictions):.4f}")
        logger.info(f"  Std: {np.std(predictions):.4f}")
        logger.info(f"  Min: {np.min(predictions):.4f}")
        logger.info(f"  Max: {np.max(predictions):.4f}")
        
        # Distribution analysis
        ai_ratio = sum(1 for p in predictions if p > 0.5) / len(predictions)
        logger.info(f"  Predicted AI ratio: {ai_ratio:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Ensemble prediction failed: {str(e)}")
        logger.exception("Full traceback:")


if __name__ == "__main__":
    main()