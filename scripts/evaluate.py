#!/usr/bin/env python3
"""
Evaluation Script for AI Text Detection Project
Usage: python scripts/evaluate.py [--predictions PRED_FILE] [--labels LABEL_FILE] [--detailed]
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import get_config, get_config_for_environment
from src.data_processor import DataProcessor
from src.evaluator import Evaluator
from src.utils import setup_project_management


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/evaluation.log')
        ]
    )
    Path("logs").mkdir(exist_ok=True)


def load_predictions(pred_file: str) -> tuple:
    """Load predictions from file."""
    pred_df = pd.read_csv(pred_file)
    
    if 'generated' in pred_df.columns:
        predictions = pred_df['generated'].tolist()
        ids = pred_df.get('ID', list(range(len(predictions))))
    else:
        raise ValueError(f"Predictions file {pred_file} must contain 'generated' column")
    
    return predictions, ids


def load_ground_truth(label_file: str) -> tuple:
    """Load ground truth labels from file."""
    label_df = pd.read_csv(label_file, encoding='utf-8-sig')
    
    if 'generated' in label_df.columns:
        labels = label_df['generated'].tolist()
        ids = label_df.get('ID', list(range(len(labels))))
    else:
        raise ValueError(f"Labels file {label_file} must contain 'generated' column")
    
    return labels, ids


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI Text Detection Model Performance")
    parser.add_argument("--predictions", "-p", required=True,
                       help="Predictions file (CSV with 'generated' column)")
    parser.add_argument("--labels", "-l", default=None,
                       help="Ground truth labels file (if available)")
    parser.add_argument("--test-data", "-t", default="data/test.csv",
                       help="Test data file for document-level analysis")
    parser.add_argument("--detailed", "-d", action="store_true",
                       help="Generate detailed evaluation report")
    parser.add_argument("--plots", action="store_true",
                       help="Generate evaluation plots")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Classification threshold")
    parser.add_argument("--output-dir", "-o", default="results",
                       help="Output directory for results")
    parser.add_argument("--env", default="default",
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
        config = get_config_for_environment(args.env)
        
        logger.info(f"Starting evaluation with environment: {args.env}")
        logger.info(f"Predictions file: {args.predictions}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        evaluator = Evaluator(config)
        
        # Setup project management (if dependencies are available)
        try:
            pm = setup_project_management({'log_level': args.log_level})
            logger.info("Project management enabled")
        except Exception as e:
            logger.warning(f"Project management not available: {e}")
            pm = None
        
        # Load predictions
        logger.info("Loading predictions...")
        predictions, pred_ids = load_predictions(args.predictions)
        logger.info(f"Loaded {len(predictions)} predictions")
        
        # Basic prediction analysis
        pred_stats = {
            'total_predictions': len(predictions),
            'mean_prediction': sum(predictions) / len(predictions) if predictions else 0,
            'predictions_above_threshold': sum(1 for p in predictions if p > args.threshold),
            'predictions_below_threshold': sum(1 for p in predictions if p <= args.threshold),
            'min_prediction': min(predictions) if predictions else 0,
            'max_prediction': max(predictions) if predictions else 0
        }
        
        logger.info(f"Prediction statistics: {pred_stats}")
        
        evaluation_results = {
            'prediction_file': args.predictions,
            'threshold': args.threshold,
            'prediction_statistics': pred_stats
        }
        
        # If ground truth labels are provided, calculate performance metrics
        if args.labels:
            logger.info("Loading ground truth labels...")
            labels, label_ids = load_ground_truth(args.labels)
            logger.info(f"Loaded {len(labels)} labels")
            
            # Ensure predictions and labels are aligned
            if len(predictions) != len(labels):
                logger.warning(f"Prediction count ({len(predictions)}) != label count ({len(labels)})")
                min_len = min(len(predictions), len(labels))
                predictions = predictions[:min_len]
                labels = labels[:min_len]
                logger.info(f"Truncated to {min_len} samples")
            
            # Calculate performance metrics
            logger.info("Calculating performance metrics...")
            metrics = evaluator.evaluate_model(labels, predictions, args.threshold)
            
            logger.info(f"Performance metrics: {metrics}")
            
            # Add metrics to results
            evaluation_results.update({
                'labels_file': args.labels,
                'performance_metrics': metrics,
                'evaluation_with_ground_truth': True
            })
            
            # Threshold analysis
            if args.detailed:
                logger.info("Performing threshold analysis...")
                threshold_analysis = evaluator.threshold_analysis(labels, predictions)
                evaluation_results['threshold_analysis'] = threshold_analysis
                
                logger.info(f"Best threshold: {threshold_analysis['best_threshold']:.3f} "
                          f"(F1: {threshold_analysis['best_f1_score']:.4f})")
            
            # Generate plots if requested
            if args.plots:
                logger.info("Generating evaluation plots...")
                plot_dir = Path(args.output_dir) / "plots"
                evaluator.create_evaluation_plots(labels, predictions, str(plot_dir))
            
            # Print summary
            summary = evaluator.get_model_summary(metrics)
            print("\n" + summary)
            
        else:
            logger.info("No ground truth labels provided - performing prediction analysis only")
            evaluation_results['evaluation_with_ground_truth'] = False
        
        # Document-level analysis if test data is available
        if Path(args.test_data).exists():
            logger.info("Performing document-level analysis...")
            try:
                data_processor = DataProcessor(config.data, config.model)
                _, test_df = data_processor.load_data()
                
                if len(test_df) == len(predictions):
                    doc_analysis = evaluator.analyze_document_level_performance(test_df, predictions)
                    evaluation_results['document_analysis'] = doc_analysis
                    logger.info(f"Document-level analysis: {doc_analysis}")
                else:
                    logger.warning(f"Test data length ({len(test_df)}) != predictions length ({len(predictions)})")
            
            except Exception as e:
                logger.warning(f"Document-level analysis failed: {e}")
        
        # Save evaluation report
        report_file = Path(args.output_dir) / "evaluation_report.json"
        evaluator.save_evaluation_report(evaluation_results, str(report_file))
        
        # Save detailed results if requested
        if args.detailed:
            detailed_file = Path(args.output_dir) / "detailed_evaluation.json"
            
            detailed_results = evaluation_results.copy()
            detailed_results.update({
                'predictions': predictions,
                'prediction_ids': pred_ids,
                'configuration': config.to_dict()
            })
            
            if args.labels:
                detailed_results.update({
                    'labels': labels,
                    'label_ids': label_ids
                })
            
            evaluator.save_evaluation_report(detailed_results, str(detailed_file))
            logger.info(f"Detailed results saved to {detailed_file}")
        
        logger.info("Evaluation completed successfully!")
        
        # Auto-commit if project management is available
        if pm:
            try:
                if args.labels:
                    auc_score = evaluation_results['performance_metrics']['auc_score']
                    commit_msg = f"Complete model evaluation with AUC: {auc_score:.4f}"
                else:
                    commit_msg = "Complete prediction analysis without ground truth"
                
                pm.auto_workflow("evaluation_completed", commit_msg)
            except Exception as e:
                logger.warning(f"Auto-commit failed: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())