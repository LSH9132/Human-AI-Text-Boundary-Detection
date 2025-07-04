#!/usr/bin/env python3
"""
Training Script for AI Text Detection Project
Usage: python scripts/train.py [--config CONFIG] [--env ENV]
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import get_config, get_config_for_environment
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator
from src.utils import setup_project_management


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/training.log')
        ]
    )
    Path("logs").mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Train AI Text Detection Model")
    parser.add_argument("--config", "-c", default=None, help="Config file path")
    parser.add_argument("--env", "-e", default="default", 
                       choices=["default", "gpu", "cpu", "debug"],
                       help="Environment configuration")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if args.config:
            # TODO: Implement config file loading
            config = get_config()
            logger.info(f"Custom config loading not implemented yet, using default")
        else:
            config = get_config_for_environment(args.env)
        
        logger.info(f"Starting training with environment: {args.env}")
        logger.info(f"Device: {config.system.device}")
        logger.info(f"Batch size: {config.training.batch_size}")
        logger.info(f"Learning rate: {config.training.learning_rate}")
        
        # Initialize components
        data_processor = DataProcessor(config.data, config.model)
        model_trainer = ModelTrainer(config)
        evaluator = Evaluator(config)
        
        # Setup project management (if dependencies are available)
        try:
            pm = setup_project_management({'log_level': args.log_level})
            logger.info("Project management enabled")
        except Exception as e:
            logger.warning(f"Project management not available: {e}")
            pm = None
        
        # Load and preprocess data
        logger.info("Loading data...")
        train_df, test_df = data_processor.load_data()
        
        # Validate data integrity
        if not data_processor.validate_data_integrity(train_df, test_df):
            logger.error("Data validation failed")
            return 1
        
        # Log data statistics
        stats = data_processor.get_data_statistics(train_df, test_df)
        logger.info(f"Data statistics: {stats}")
        
        # Preprocess training data
        train_para_df = data_processor.preprocess_training_data(train_df)
        
        # Log class distribution
        class_dist = data_processor.get_class_distribution(train_para_df['generated'].tolist())
        logger.info(f"Class distribution: {class_dist}")
        
        # Prepare training data
        train_texts = train_para_df['paragraph_text'].tolist()
        train_labels = train_para_df['generated'].tolist()
        
        logger.info(f"Training data prepared: {len(train_texts)} samples")
        
        # Perform cross-validation training
        logger.info("Starting cross-validation training...")
        oof_auc, model_paths = model_trainer.cross_validate(train_texts, train_labels)
        
        logger.info(f"Training completed! OOF AUC: {oof_auc:.4f}")
        logger.info(f"Model checkpoints saved: {model_paths}")
        
        # Save training metrics
        training_metrics = {
            'oof_auc': oof_auc,
            'model_paths': model_paths,
            'config': config.to_dict(),
            'data_stats': stats,
            'class_distribution': class_dist
        }
        
        model_trainer.save_training_metrics(training_metrics)
        
        # Log model information
        model_info = model_trainer.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        # Create evaluation report
        evaluation_metrics = {
            'training_completed': True,
            'oof_auc': oof_auc,
            'n_folds': config.training.n_splits,
            'model_paths': model_paths,
            'training_samples': len(train_texts),
            'model_info': model_info
        }
        
        evaluator.save_evaluation_report(evaluation_metrics, "training_report.json")
        
        logger.info("Training pipeline completed successfully!")
        
        # Auto-commit if project management is available
        if pm:
            try:
                pm.auto_workflow("training_completed", 
                               f"Complete model training with OOF AUC: {oof_auc:.4f}")
            except Exception as e:
                logger.warning(f"Auto-commit failed: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())