"""
Model training module for AI Text Detection project.
Handles model initialization, training, validation, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional
import os
from tqdm import tqdm

from .config import Config
from .data_processor import TextDataset


class ModelTrainer:
    """Main model training class."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(config.system.device)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        
    def create_model(self) -> AutoModelForSequenceClassification:
        """Create and initialize the model."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.model_name,
            num_labels=self.config.model.num_labels
        )
        return model.to(self.device)
    
    def create_optimizer_and_scheduler(self, model: nn.Module, 
                                     total_steps: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """Create optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.config.training.warmup_ratio * total_steps),
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR,
                   criterion: nn.Module) -> float:
        """Train model for one epoch with gradient accumulation."""
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        # Get gradient accumulation steps from config
        gradient_accumulation_steps = getattr(self.config.training, 'gradient_accumulation_steps', 1)
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
        
        # Final update if needed
        if len(train_loader) % gradient_accumulation_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader) -> Tuple[float, List[float], List[float]]:
        """Validate model for one epoch."""
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].cpu().numpy()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
                
                # Handle both scalar and array predictions
                if preds.ndim == 0:
                    val_preds.append(preds.item())
                else:
                    val_preds.extend(preds)
                val_labels.extend(labels)
        
        auc_score = roc_auc_score(val_labels, val_preds)
        return auc_score, val_preds, val_labels
    
    def train_single_fold(self, train_dataset: TextDataset, val_dataset: TextDataset,
                         fold: int) -> Tuple[float, str]:
        """Train model for a single fold."""
        self.logger.info(f"Training fold {fold + 1}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.system.num_workers,
            pin_memory=self.config.system.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.system.num_workers,
            pin_memory=self.config.system.pin_memory
        )
        
        # Initialize model
        model = self.create_model()
        
        # Setup training components
        # Adjust total steps for gradient accumulation
        gradient_accumulation_steps = getattr(self.config.training, 'gradient_accumulation_steps', 1)
        total_steps = (len(train_loader) // gradient_accumulation_steps) * self.config.training.epochs
        optimizer, scheduler = self.create_optimizer_and_scheduler(model, total_steps)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_auc = 0
        patience_counter = 0
        best_model_path = None
        
        for epoch in range(self.config.training.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler, criterion)
            
            # Validate
            val_auc, _, _ = self.validate_epoch(model, val_loader)
            
            self.logger.info(f"Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                
                # Save model checkpoint
                model_filename = self.config.system.checkpoint_pattern.format(fold=fold + 1)
                model_path = os.path.join(self.config.system.model_save_dir, model_filename)
                torch.save(model.state_dict(), model_path)
                best_model_path = model_path
                
                self.logger.info(f"New best model saved: {model_path}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.config.training.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Clear GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return best_auc, best_model_path
    
    def cross_validate(self, train_data: List[str], train_labels: List[int]) -> Tuple[float, List[str]]:
        """Perform cross-validation training."""
        self.logger.info(f"Starting {self.config.training.n_splits}-fold cross-validation")
        
        # Setup cross-validation
        skf = StratifiedKFold(n_splits=self.config.training.n_splits, shuffle=True, random_state=42)
        
        fold_scores = []
        model_paths = []
        oof_predictions = np.zeros(len(train_data))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, train_labels)):
            self.logger.info(f"\n--- Fold {fold + 1}/{self.config.training.n_splits} ---")
            
            # Split data
            fold_train_texts = [train_data[i] for i in train_idx]
            fold_train_labels = [train_labels[i] for i in train_idx]
            fold_val_texts = [train_data[i] for i in val_idx]
            fold_val_labels = [train_labels[i] for i in val_idx]
            
            # Create datasets
            train_dataset = TextDataset(fold_train_texts, fold_train_labels, self.tokenizer, 
                                      self.config.model.max_length)
            val_dataset = TextDataset(fold_val_texts, fold_val_labels, self.tokenizer, 
                                    self.config.model.max_length)
            
            # Train fold
            fold_auc, model_path = self.train_single_fold(train_dataset, val_dataset, fold)
            
            # Generate OOF predictions
            model = self.create_model()
            model.load_state_dict(torch.load(model_path))
            val_loader = DataLoader(val_dataset, batch_size=self.config.training.batch_size, 
                                  shuffle=False)
            
            _, oof_preds, _ = self.validate_epoch(model, val_loader)
            oof_predictions[val_idx] = oof_preds
            
            fold_scores.append(fold_auc)
            model_paths.append(model_path)
            
            self.logger.info(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
            
            # Clear GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Calculate overall OOF score
        oof_auc = roc_auc_score(train_labels, oof_predictions)
        
        self.logger.info(f"\nCross-validation completed!")
        self.logger.info(f"Fold scores: {[f'{score:.4f}' for score in fold_scores]}")
        self.logger.info(f"Mean CV AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        self.logger.info(f"OOF AUC: {oof_auc:.4f}")
        
        return oof_auc, model_paths
    
    def load_model(self, model_path: str) -> AutoModelForSequenceClassification:
        """Load a trained model from checkpoint."""
        model = self.create_model()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def save_training_metrics(self, metrics: Dict[str, float], filename: str = "training_metrics.json") -> None:
        """Save training metrics to file."""
        import json
        
        metrics_path = os.path.join(self.config.system.model_save_dir, filename)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Training metrics saved to {metrics_path}")
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model architecture information."""
        model = self.create_model()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.config.model.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # Assuming float32
            'max_sequence_length': self.config.model.max_length,
            'num_labels': self.config.model.num_labels
        }