#!/usr/bin/env python3
"""
Prediction Script for AI Text Detection Project
Usage: python scripts/predict.py [--models MODEL_PATHS] [--ensemble] [--output OUTPUT]
"""

import sys
import argparse
import logging
from pathlib import Path
import glob

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import get_config, get_config_for_environment
from src.data_processor import DataProcessor
from src.predictor import Predictor
from src.evaluator import Evaluator
from src.utils import setup_project_management


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/prediction.log')
        ]
    )
    Path("logs").mkdir(exist_ok=True)


def find_model_files(pattern: str = "models/best_model_fold_*.pt") -> list:
    """Find model files matching the pattern."""
    model_files = glob.glob(pattern)
    model_files.sort()  # Ensure consistent ordering
    return model_files


def main():
    parser = argparse.ArgumentParser(description="Generate Predictions for AI Text Detection")
    parser.add_argument("--models", "-m", nargs="*", default=None,
                       help="Model file paths (if not provided, auto-discover)")
    parser.add_argument("--ensemble", "-e", action="store_true",
                       help="Use ensemble prediction (default: True if multiple models)")
    parser.add_argument("--output", "-o", default="submission.csv",
                       help="Output file name")
    parser.add_argument("--method", default="mean",
                       choices=["mean", "median", "weighted_mean"],
                       help="Ensemble method")
    parser.add_argument("--env", default="default",
                       choices=["default", "gpu", "cpu", "debug"],
                       help="Environment configuration")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--save-detailed", action="store_true",
                       help="Save detailed predictions with metadata")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = get_config_for_environment(args.env)
        
        logger.info(f"Starting prediction with environment: {args.env}")
        logger.info(f"Device: {config.system.device}")
        logger.info(f"Output file: {args.output}")
        
        # Initialize components
        data_processor = DataProcessor(config.data, config.model)
        predictor = Predictor(config)
        evaluator = Evaluator(config)
        
        # Setup project management (if dependencies are available)
        try:
            pm = setup_project_management({'log_level': args.log_level})
            logger.info("Project management enabled")
        except Exception as e:
            logger.warning(f"Project management not available: {e}")
            pm = None
        
        # Find model files
        if args.models:
            model_paths = args.models
            logger.info(f"Using provided model paths: {model_paths}")
        else:
            model_paths = find_model_files()
            logger.info(f"Auto-discovered model paths: {model_paths}")
        
        if not model_paths:
            logger.error("No model files found. Please train models first or provide model paths.")
            return 1
        
        # Validate model files exist
        missing_models = [p for p in model_paths if not Path(p).exists()]
        if missing_models:
            logger.error(f"Model files not found: {missing_models}")
            return 1
        
        # Load test data
        logger.info("Loading test data...")
        _, test_df = data_processor.load_data()
        
        logger.info(f"Test data loaded: {len(test_df)} samples")
        
        # Determine prediction method
        use_ensemble = args.ensemble or len(model_paths) > 1
        
        if use_ensemble:
            logger.info(f"Using ensemble prediction with {len(model_paths)} models")
            logger.info(f"Ensemble method: {args.method}")
            
            # Generate ensemble predictions
            predictions = predictor.ensemble_predict(
                model_paths, test_df, method=args.method
            )
            
        else:
            logger.info(f"Using single model prediction: {model_paths[0]}")
            
            # Generate single model predictions
            predictions = predictor.predict_single_model(model_paths[0], test_df)
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Analyze predictions
        pred_analysis = predictor.analyze_predictions(predictions, test_df)
        logger.info(f"Prediction analysis: {pred_analysis}")
        
        # Document-level analysis
        doc_analysis = evaluator.analyze_document_level_performance(test_df, predictions)
        logger.info(f"Document-level analysis: {doc_analysis}")
        
        # Prepare submission
        submission_df = data_processor.prepare_submission_format(test_df, predictions)
        
        # Update output path in config
        config.data.submission_file = args.output
        
        # Save submission
        data_processor.save_submission(submission_df)
        logger.info(f"Submission saved to {args.output}")
        
        # Save detailed predictions if requested
        if args.save_detailed:
            predictor.save_detailed_predictions(predictions, test_df)
            logger.info("Detailed predictions saved")
        
        # Create evaluation report
        prediction_report = {
            'prediction_completed': True,
            'model_paths': model_paths,
            'ensemble_used': use_ensemble,
            'ensemble_method': args.method if use_ensemble else None,
            'total_predictions': len(predictions),
            'prediction_analysis': pred_analysis,
            'document_analysis': doc_analysis,
            'output_file': args.output
        }
        
        evaluator.save_evaluation_report(prediction_report, "prediction_report.json")
        
        logger.info("Prediction pipeline completed successfully!")
        
        # Auto-commit if project management is available
        if pm:
            try:
                pm.auto_workflow("prediction_completed", 
                               f"Generate predictions using {'ensemble' if use_ensemble else 'single'} model")
            except Exception as e:
                logger.warning(f"Auto-commit failed: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())