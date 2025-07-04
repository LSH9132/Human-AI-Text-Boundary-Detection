"""
Prediction module for AI Text Detection project.
Handles model inference, context-aware prediction, and ensemble methods.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

from .config import Config
from .data_processor import TextDataset


class Predictor:
    """Main prediction class with context-aware capabilities."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(config.system.device)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    
    def load_model(self, model_path: str) -> AutoModelForSequenceClassification:
        """Load a trained model from checkpoint."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.model_name,
            num_labels=self.config.model.num_labels
        ).to(self.device)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def predict_batch(self, model: nn.Module, texts: List[str]) -> List[float]:
        """Predict probabilities for a batch of texts."""
        dataset = TextDataset(texts, None, self.tokenizer, self.config.model.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.system.num_workers,
            pin_memory=self.config.system.pin_memory
        )
        
        predictions = []
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
                
                if len(probs.shape) == 0:  # Single prediction
                    predictions.append(float(probs))
                else:
                    predictions.extend(probs.tolist())
        
        return predictions
    
    def predict_with_context(self, model: nn.Module, test_df: pd.DataFrame) -> List[float]:
        """Predict with document-level context adjustment."""
        self.logger.info("Performing context-aware prediction...")
        
        predictions = []
        grouped = test_df.groupby('title')
        
        for title, group in tqdm(grouped, desc="Processing documents"):
            # Get paragraphs for this document
            paragraphs = group['paragraph_text'].tolist()
            
            # Get individual predictions for each paragraph
            individual_preds = self.predict_batch(model, paragraphs)
            
            # Calculate document-level average
            doc_avg = np.mean(individual_preds)
            
            # Apply context adjustment
            adjusted_preds = []
            for pred in individual_preds:
                adjusted = (
                    (1 - self.config.training.context_weight) * pred + 
                    self.config.training.context_weight * doc_avg
                )
                adjusted_preds.append(adjusted)
            
            predictions.extend(adjusted_preds)
        
        return predictions
    
    def ensemble_predict(self, model_paths: List[str], test_df: pd.DataFrame,
                        method: str = 'mean') -> List[float]:
        """Ensemble prediction using multiple models."""
        self.logger.info(f"Performing ensemble prediction with {len(model_paths)} models")
        
        all_predictions = []
        
        for i, model_path in enumerate(model_paths):
            self.logger.info(f"Predicting with model {i + 1}/{len(model_paths)}")
            
            model = self.load_model(model_path)
            preds = self.predict_with_context(model, test_df)
            all_predictions.append(preds)
            
            # Clear GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Combine predictions
        all_predictions = np.array(all_predictions)
        
        if method == 'mean':
            ensemble_preds = np.mean(all_predictions, axis=0)
        elif method == 'median':
            ensemble_preds = np.median(all_predictions, axis=0)
        elif method == 'weighted_mean':
            # You can implement weighted averaging here
            weights = np.ones(len(model_paths)) / len(model_paths)
            ensemble_preds = np.average(all_predictions, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return ensemble_preds.tolist()
    
    def predict_single_model(self, model_path: str, test_df: pd.DataFrame) -> List[float]:
        """Predict using a single model."""
        self.logger.info(f"Predicting with single model: {model_path}")
        
        model = self.load_model(model_path)
        predictions = self.predict_with_context(model, test_df)
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return predictions
    
    def predict_without_context(self, model: nn.Module, test_df: pd.DataFrame) -> List[float]:
        """Predict without document-level context adjustment."""
        self.logger.info("Performing prediction without context adjustment...")
        
        texts = test_df['paragraph_text'].tolist()
        predictions = self.predict_batch(model, texts)
        
        return predictions
    
    def calibrate_predictions(self, predictions: List[float], 
                            validation_labels: Optional[List[int]] = None,
                            method: str = 'platt') -> List[float]:
        """Calibrate prediction probabilities."""
        if validation_labels is None:
            self.logger.warning("No validation labels provided for calibration")
            return predictions
        
        if method == 'platt':
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
            
            # Simple Platt scaling
            lr = LogisticRegression()
            predictions_array = np.array(predictions).reshape(-1, 1)
            lr.fit(predictions_array, validation_labels)
            
            calibrated = lr.predict_proba(predictions_array)[:, 1]
            return calibrated.tolist()
        
        elif method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(predictions, validation_labels)
            calibrated = ir.predict(predictions)
            return calibrated.tolist()
        
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def analyze_predictions(self, predictions: List[float], test_df: pd.DataFrame) -> Dict[str, any]:
        """Analyze prediction statistics."""
        predictions_array = np.array(predictions)
        
        analysis = {
            'total_predictions': len(predictions),
            'mean_prediction': float(np.mean(predictions_array)),
            'std_prediction': float(np.std(predictions_array)),
            'min_prediction': float(np.min(predictions_array)),
            'max_prediction': float(np.max(predictions_array)),
            'median_prediction': float(np.median(predictions_array)),
            'predictions_above_0.5': int(np.sum(predictions_array > 0.5)),
            'predictions_below_0.5': int(np.sum(predictions_array < 0.5))
        }
        
        # Document-level analysis
        test_df_copy = test_df.copy()
        test_df_copy['predictions'] = predictions
        
        doc_stats = test_df_copy.groupby('title')['predictions'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        analysis['documents_analyzed'] = len(doc_stats)
        analysis['avg_paragraphs_per_doc'] = float(doc_stats['count'].mean())
        analysis['doc_level_mean_pred'] = float(doc_stats['mean'].mean())
        analysis['doc_level_std_pred'] = float(doc_stats['mean'].std())
        
        return analysis
    
    def save_detailed_predictions(self, predictions: List[float], test_df: pd.DataFrame,
                                filename: str = "detailed_predictions.csv") -> None:
        """Save detailed predictions with metadata."""
        detailed_df = test_df.copy()
        detailed_df['predicted_probability'] = predictions
        detailed_df['predicted_class'] = (np.array(predictions) > 0.5).astype(int)
        
        # Add document-level statistics
        doc_stats = detailed_df.groupby('title')['predicted_probability'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        doc_stats.columns = ['title', 'doc_mean_prob', 'doc_std_prob', 'doc_paragraph_count']
        
        detailed_df = detailed_df.merge(doc_stats, on='title')
        
        output_path = f"results/{filename}"
        import os
        os.makedirs("results", exist_ok=True)
        detailed_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Detailed predictions saved to {output_path}")
    
    def predict_confidence_intervals(self, model_paths: List[str], test_df: pd.DataFrame,
                                   confidence_level: float = 0.95) -> Tuple[List[float], List[float], List[float]]:
        """Predict with confidence intervals using ensemble."""
        if len(model_paths) < 3:
            self.logger.warning("Need at least 3 models for meaningful confidence intervals")
        
        all_predictions = []
        
        for model_path in model_paths:
            model = self.load_model(model_path)
            preds = self.predict_with_context(model, test_df)
            all_predictions.append(preds)
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        all_predictions = np.array(all_predictions)
        
        # Calculate statistics
        mean_preds = np.mean(all_predictions, axis=0)
        std_preds = np.std(all_predictions, axis=0)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        z_score = 1.96  # For 95% confidence interval
        
        lower_bound = mean_preds - z_score * std_preds
        upper_bound = mean_preds + z_score * std_preds
        
        # Clip to [0, 1] range
        lower_bound = np.clip(lower_bound, 0, 1)
        upper_bound = np.clip(upper_bound, 0, 1)
        
        return mean_preds.tolist(), lower_bound.tolist(), upper_bound.tolist()