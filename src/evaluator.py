"""
Evaluation module for AI Text Detection project.
Handles metrics calculation, model evaluation, and performance analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import json
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
        classification_report, accuracy_score, precision_score, recall_score, f1_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Some evaluation features may be limited.")

from .config import Config


class Evaluator:
    """Main evaluation class for model performance assessment."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available. Using basic evaluation metrics only.")
    
    def calculate_basic_metrics(self, y_true: List[int], y_pred: List[float], 
                               threshold: float = 0.5) -> Dict[str, float]:
        """Calculate basic classification metrics without sklearn."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Basic calculations
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        
        # Avoid division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def calculate_auc_manual(self, y_true: List[int], y_pred: List[float]) -> float:
        """Calculate AUC manually without sklearn."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Sort by prediction scores
        sorted_indices = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Calculate AUC using trapezoidal rule
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        tpr_points = []
        fpr_points = []
        
        tp = fp = 0
        for i, label in enumerate(y_true_sorted):
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            tpr = tp / n_pos
            fpr = fp / n_neg
            tpr_points.append(tpr)
            fpr_points.append(fpr)
        
        # Calculate AUC using trapezoidal integration
        auc = 0.0
        for i in range(1, len(fpr_points)):
            auc += (fpr_points[i] - fpr_points[i-1]) * (tpr_points[i] + tpr_points[i-1]) / 2
        
        return auc
    
    def evaluate_model(self, y_true: List[int], y_pred: List[float], 
                      threshold: float = 0.5) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        self.logger.info("Evaluating model performance...")
        
        metrics = {}
        
        # Basic metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred, threshold)
        metrics.update(basic_metrics)
        
        # AUC calculation
        if SKLEARN_AVAILABLE:
            try:
                auc_score = roc_auc_score(y_true, y_pred)
                metrics['auc_score'] = auc_score
            except Exception as e:
                self.logger.warning(f"sklearn AUC calculation failed: {e}, using manual calculation")
                metrics['auc_score'] = self.calculate_auc_manual(y_true, y_pred)
        else:
            metrics['auc_score'] = self.calculate_auc_manual(y_true, y_pred)
        
        # Additional statistics
        y_pred_array = np.array(y_pred)
        metrics.update({
            'mean_prediction': float(np.mean(y_pred_array)),
            'std_prediction': float(np.std(y_pred_array)),
            'min_prediction': float(np.min(y_pred_array)),
            'max_prediction': float(np.max(y_pred_array)),
            'threshold_used': threshold,
            'total_samples': len(y_true),
            'positive_samples': int(np.sum(y_true)),
            'negative_samples': int(len(y_true) - np.sum(y_true))
        })
        
        self.logger.info(f"Evaluation completed. AUC: {metrics['auc_score']:.4f}, "
                        f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def evaluate_cross_validation(self, cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate cross-validation results."""
        self.logger.info("Evaluating cross-validation results...")
        
        # Extract metrics from each fold
        fold_metrics = []
        for fold_result in cv_results:
            fold_metrics.append(fold_result)
        
        # Calculate statistics across folds
        metrics_summary = {}
        
        if fold_metrics:
            # Get all metric names
            metric_names = set()
            for fold in fold_metrics:
                metric_names.update(fold.keys())
            
            # Calculate mean and std for each metric
            for metric in metric_names:
                values = [fold.get(metric, 0) for fold in fold_metrics]
                if all(isinstance(v, (int, float)) for v in values):
                    metrics_summary[f'{metric}_mean'] = float(np.mean(values))
                    metrics_summary[f'{metric}_std'] = float(np.std(values))
                    metrics_summary[f'{metric}_min'] = float(np.min(values))
                    metrics_summary[f'{metric}_max'] = float(np.max(values))
        
        metrics_summary['n_folds'] = len(fold_metrics)
        metrics_summary['fold_results'] = fold_metrics
        
        return metrics_summary
    
    def threshold_analysis(self, y_true: List[int], y_pred: List[float], 
                          thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
        """Analyze performance across different thresholds."""
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1).tolist()
        
        self.logger.info(f"Analyzing performance across {len(thresholds)} thresholds")
        
        threshold_results = []
        
        for threshold in thresholds:
            metrics = self.calculate_basic_metrics(y_true, y_pred, threshold)
            metrics['threshold'] = threshold
            threshold_results.append(metrics)
        
        # Find best threshold based on F1 score
        best_f1_result = max(threshold_results, key=lambda x: x['f1_score'])
        
        return {
            'threshold_results': threshold_results,
            'best_threshold': best_f1_result['threshold'],
            'best_f1_score': best_f1_result['f1_score'],
            'best_metrics': best_f1_result
        }
    
    def analyze_document_level_performance(self, test_df: pd.DataFrame, 
                                         predictions: List[float]) -> Dict[str, Any]:
        """Analyze performance at document level."""
        self.logger.info("Analyzing document-level performance...")
        
        # Add predictions to dataframe
        analysis_df = test_df.copy()
        analysis_df['predictions'] = predictions
        
        # Group by document (title)
        doc_stats = analysis_df.groupby('title').agg({
            'predictions': ['mean', 'std', 'count', 'min', 'max'],
            'paragraph_index': 'max'
        }).reset_index()
        
        doc_stats.columns = ['title', 'mean_pred', 'std_pred', 'paragraph_count', 
                           'min_pred', 'max_pred', 'max_paragraph_index']
        
        # Calculate document-level statistics
        analysis = {
            'total_documents': len(doc_stats),
            'total_paragraphs': len(analysis_df),
            'avg_paragraphs_per_document': float(doc_stats['paragraph_count'].mean()),
            'std_paragraphs_per_document': float(doc_stats['paragraph_count'].std()),
            'min_paragraphs_per_document': int(doc_stats['paragraph_count'].min()),
            'max_paragraphs_per_document': int(doc_stats['paragraph_count'].max()),
            'avg_document_prediction': float(doc_stats['mean_pred'].mean()),
            'std_document_prediction': float(doc_stats['mean_pred'].std()),
            'documents_above_threshold': int(np.sum(doc_stats['mean_pred'] > 0.5)),
            'documents_below_threshold': int(np.sum(doc_stats['mean_pred'] <= 0.5))
        }
        
        return analysis
    
    def save_evaluation_report(self, metrics: Dict[str, Any], 
                              output_path: str = "evaluation_report.json") -> None:
        """Save evaluation results to file."""
        # Create results directory
        Path("results").mkdir(exist_ok=True)
        
        full_path = Path("results") / output_path
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_metrics = self._convert_to_serializable(metrics)
        
        with open(full_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        self.logger.info(f"Evaluation report saved to {full_path}")
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    def create_evaluation_plots(self, y_true: List[int], y_pred: List[float],
                               output_dir: str = "results/plots") -> None:
        """Create evaluation plots (if matplotlib is available)."""
        if not PLOTTING_AVAILABLE:
            self.logger.warning("matplotlib/seaborn not available. Skipping plot generation.")
            return
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Prediction distribution
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Probabilities')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/prediction_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve (if sklearn is available)
        if SKLEARN_AVAILABLE:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                auc_score = roc_auc_score(y_true, y_pred)
                
                plt.figure(figsize=(8, 8))
                plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{output_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                self.logger.warning(f"Failed to create ROC curve: {e}")
        
        self.logger.info(f"Evaluation plots saved to {output_dir}")
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance of multiple models."""
        self.logger.info(f"Comparing {len(model_results)} models...")
        
        comparison = {
            'models': list(model_results.keys()),
            'metrics_comparison': {},
            'best_model': {},
            'rankings': {}
        }
        
        # Extract common metrics
        common_metrics = set()
        for model_metrics in model_results.values():
            common_metrics.update(model_metrics.keys())
        
        # Compare each metric
        for metric in common_metrics:
            metric_values = {}
            for model_name, metrics in model_results.items():
                if metric in metrics and isinstance(metrics[metric], (int, float)):
                    metric_values[model_name] = metrics[metric]
            
            if metric_values:
                comparison['metrics_comparison'][metric] = metric_values
                
                # Find best model for this metric
                best_model = max(metric_values.items(), key=lambda x: x[1])
                comparison['best_model'][metric] = {
                    'model': best_model[0],
                    'value': best_model[1]
                }
        
        return comparison
    
    def get_model_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable model summary."""
        summary_lines = [
            "=== Model Performance Summary ===",
            f"AUC Score: {metrics.get('auc_score', 'N/A'):.4f}",
            f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}",
            f"Precision: {metrics.get('precision', 'N/A'):.4f}",
            f"Recall: {metrics.get('recall', 'N/A'):.4f}",
            f"F1 Score: {metrics.get('f1_score', 'N/A'):.4f}",
            "",
            f"Total Samples: {metrics.get('total_samples', 'N/A')}",
            f"Positive Samples: {metrics.get('positive_samples', 'N/A')}",
            f"Negative Samples: {metrics.get('negative_samples', 'N/A')}",
            "",
            f"Mean Prediction: {metrics.get('mean_prediction', 'N/A'):.4f}",
            f"Std Prediction: {metrics.get('std_prediction', 'N/A'):.4f}",
            "================================="
        ]
        
        return "\n".join(summary_lines)