#!/usr/bin/env python3
"""
AI Text Detection - Main Execution Script
Modernized version using modular architecture

Usage:
    python main.py                  # Default training and prediction
    python main.py --env gpu        # Use GPU configuration  
    python main.py --env cpu        # Use CPU configuration
    python main.py --env debug      # Debug mode (faster for testing)
"""

import sys
import argparse
import logging
from pathlib import Path
import warnings

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import modular components
try:
    from src.config import get_config_for_environment
    from src.data_processor import DataProcessor
    from src.model_trainer import ModelTrainer
    from src.predictor import Predictor
    from src.evaluator import Evaluator
    from src.utils import setup_project_management
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Modular components not available: {e}")
    print("Falling back to legacy implementation...")
    MODULES_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/main.log')
        ]
    )


def run_modular_pipeline(config_env: str = "default", log_level: str = "INFO"):
    """Run the modular ML pipeline."""
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== AI Text Detection - Modular Pipeline ===")
        logger.info(f"Environment: {config_env}")
        
        # Load configuration
        config = get_config_for_environment(config_env)
        logger.info(f"Device: {config.system.device}")
        logger.info(f"Model: {config.model.model_name}")
        logger.info(f"Batch size: {config.training.batch_size}")
        
        # Initialize components
        logger.info("Initializing components...")
        data_processor = DataProcessor(config.data, config.model)
        model_trainer = ModelTrainer(config)
        predictor = Predictor(config)
        evaluator = Evaluator(config)
        
        # Setup project management (optional)
        try:
            pm = setup_project_management({'log_level': log_level})
            logger.info("Project management enabled")
        except Exception as e:
            logger.warning(f"Project management not available: {e}")
            pm = None
        
        # === 1. DATA PROCESSING ===
        logger.info("\n=== 1. DATA PROCESSING ===")
        
        # Load data
        train_df, test_df = data_processor.load_data()
        
        # Validate data
        if not data_processor.validate_data_integrity(train_df, test_df):
            raise ValueError("Data validation failed")
        
        # Log data statistics
        stats = data_processor.get_data_statistics(train_df, test_df)
        logger.info(f"Training samples: {stats['train_samples']:,}")
        logger.info(f"Test samples: {stats['test_samples']:,}")
        
        # Preprocess training data (split into paragraphs)
        train_para_df = data_processor.preprocess_training_data(train_df)
        
        # Log class distribution
        class_dist = data_processor.get_class_distribution(train_para_df['generated'].tolist())
        logger.info(f"Class distribution: {class_dist}")
        
        # === 2. MODEL TRAINING ===
        logger.info("\n=== 2. MODEL TRAINING ===")
        
        # Prepare training data
        train_texts = train_para_df['paragraph_text'].tolist()
        train_labels = train_para_df['generated'].tolist()
        
        logger.info(f"Training on {len(train_texts):,} paragraph samples")
        
        # Perform cross-validation training
        oof_auc, model_paths = model_trainer.cross_validate(train_texts, train_labels)
        
        logger.info(f"Training completed!")
        logger.info(f"Out-of-fold AUC: {oof_auc:.4f}")
        logger.info(f"Model checkpoints: {len(model_paths)} folds")
        
        # === 3. PREDICTION ===
        logger.info("\n=== 3. PREDICTION ===")
        
        # Generate ensemble predictions
        predictions = predictor.ensemble_predict(model_paths, test_df, method='mean')
        
        logger.info(f"Generated {len(predictions):,} predictions")
        
        # Analyze predictions
        pred_analysis = predictor.analyze_predictions(predictions, test_df)
        logger.info(f"Mean prediction: {pred_analysis['mean_prediction']:.4f}")
        logger.info(f"Predictions > 0.5: {pred_analysis['predictions_above_0.5']:,}")
        
        # === 4. SAVE RESULTS ===
        logger.info("\n=== 4. SAVE RESULTS ===")
        
        # Prepare and save submission
        submission_df = data_processor.prepare_submission_format(test_df, predictions)
        data_processor.save_submission(submission_df)
        
        logger.info(f"Submission saved to: {config.data.submission_file}")
        
        # === 5. EVALUATION & REPORTING ===
        logger.info("\n=== 5. EVALUATION & REPORTING ===")
        
        # Document-level analysis
        doc_analysis = evaluator.analyze_document_level_performance(test_df, predictions)
        logger.info(f"Documents analyzed: {doc_analysis['total_documents']}")
        logger.info(f"Avg paragraphs per doc: {doc_analysis['avg_paragraphs_per_document']:.1f}")
        
        # Save comprehensive report
        final_report = {
            'pipeline_completed': True,
            'environment': config_env,
            'oof_auc': oof_auc,
            'model_paths': model_paths,
            'data_statistics': stats,
            'class_distribution': class_dist,
            'prediction_analysis': pred_analysis,
            'document_analysis': doc_analysis,
            'config': config.to_dict()
        }
        
        evaluator.save_evaluation_report(final_report, "pipeline_report.json")
        
        # Save training metrics
        model_trainer.save_training_metrics({
            'oof_auc': oof_auc,
            'model_paths': model_paths,
            'final_report': final_report
        })
        
        logger.info("\n=== PIPELINE COMPLETED SUCCESSFULLY! ===")
        logger.info(f"✅ Out-of-fold AUC: {oof_auc:.4f}")
        logger.info(f"✅ Submission file: {config.data.submission_file}")
        logger.info(f"✅ Models saved: {len(model_paths)} checkpoints")
        logger.info(f"✅ Results saved to: results/")
        
        # Auto-commit if project management is available
        if pm:
            try:
                pm.auto_workflow("pipeline_completed", 
                               f"Complete full ML pipeline with OOF AUC: {oof_auc:.4f}")
                logger.info("✅ Changes committed to Git")
            except Exception as e:
                logger.warning(f"Auto-commit failed: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


def run_legacy_pipeline():
    """Fallback to legacy implementation if modular components fail."""
    print("Running legacy pipeline...")
    print("Please ensure all dependencies are installed:")
    print("pip install transformers torch pandas numpy scikit-learn")
    
    try:
        # Import and run legacy main
        sys.path.append(str(Path(__file__).parent))
        from main_legacy import main as legacy_main
        return legacy_main()
    except Exception as e:
        print(f"Legacy pipeline also failed: {e}")
        print("Please check your Python environment and dependencies.")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Text Detection Pipeline")
    parser.add_argument("--env", "-e", default="default",
                       choices=["default", "gpu", "cpu", "debug"],
                       help="Environment configuration")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--legacy", action="store_true",
                       help="Force use of legacy implementation")
    
    args = parser.parse_args()
    
    # Show banner
    print("🤖 AI Text Detection - Korean BERT Classification")
    print("=" * 50)
    
    # Choose implementation
    if args.legacy or not MODULES_AVAILABLE:
        return run_legacy_pipeline()
    else:
        return run_modular_pipeline(args.env, args.log_level)


if __name__ == "__main__":
    sys.exit(main())