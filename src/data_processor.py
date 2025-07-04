"""
Data processing module for AI Text Detection project.
Handles data loading, preprocessing, and paragraph segmentation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import logging

from .config import DataConfig, ModelConfig


class TextDataset(Dataset):
    """Custom dataset class for text classification."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, 
                 tokenizer: AutoTokenizer = None, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return item


class DataProcessor:
    """Main data processing class."""
    
    def __init__(self, data_config: DataConfig, model_config: ModelConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data from CSV files."""
        self.logger.info("Loading data files...")
        
        try:
            train_df = pd.read_csv(
                self.data_config.train_file, 
                encoding=self.data_config.encoding
            )
            test_df = pd.read_csv(
                self.data_config.test_file, 
                encoding=self.data_config.encoding
            )
            
            self.logger.info(f"Loaded {len(train_df)} training samples")
            self.logger.info(f"Loaded {len(test_df)} test samples")
            
            return train_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Filter out very short paragraphs
        paragraphs = [
            p for p in paragraphs 
            if len(p) >= self.data_config.min_paragraph_length
        ]
        
        # Limit number of paragraphs per document
        if len(paragraphs) > self.data_config.max_paragraphs_per_document:
            paragraphs = paragraphs[:self.data_config.max_paragraphs_per_document]
        
        return paragraphs
    
    def preprocess_training_data(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess training data by splitting into paragraphs."""
        self.logger.info("Preprocessing training data - splitting into paragraphs...")
        
        train_paragraphs = []
        
        for idx, row in train_df.iterrows():
            if idx % 10000 == 0:
                self.logger.info(f"Processing: {idx}/{len(train_df)}")
            
            title = row['title']
            full_text = row['full_text']
            label = row['generated']
            
            paragraphs = self.split_into_paragraphs(full_text)
            
            for p_idx, paragraph in enumerate(paragraphs):
                train_paragraphs.append({
                    'title': title,
                    'paragraph_index': p_idx,
                    'paragraph_text': paragraph,
                    'generated': label,
                    'original_idx': idx
                })
        
        train_para_df = pd.DataFrame(train_paragraphs)
        
        self.logger.info(f"Created {len(train_para_df)} paragraph samples from {len(train_df)} documents")
        
        return train_para_df
    
    def create_dataset(self, texts: List[str], labels: Optional[List[int]] = None,
                      tokenizer: AutoTokenizer = None) -> TextDataset:
        """Create a TextDataset instance."""
        return TextDataset(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            max_length=self.model_config.max_length
        )
    
    def get_class_distribution(self, labels: List[int]) -> Dict[str, float]:
        """Get class distribution statistics."""
        labels_array = np.array(labels)
        unique, counts = np.unique(labels_array, return_counts=True)
        
        distribution = {}
        total = len(labels)
        
        for label, count in zip(unique, counts):
            distribution[f'class_{int(label)}'] = count / total
        
        return distribution
    
    def validate_data_integrity(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Validate data integrity and format."""
        self.logger.info("Validating data integrity...")
        
        # Check required columns
        required_train_cols = ['title', 'full_text', 'generated']
        required_test_cols = ['ID', 'title', 'paragraph_index', 'paragraph_text']
        
        if not all(col in train_df.columns for col in required_train_cols):
            self.logger.error(f"Missing required columns in training data: {required_train_cols}")
            return False
        
        if not all(col in test_df.columns for col in required_test_cols):
            self.logger.error(f"Missing required columns in test data: {required_test_cols}")
            return False
        
        # Check for missing values
        train_nulls = train_df.isnull().sum().sum()
        test_nulls = test_df.isnull().sum().sum()
        
        if train_nulls > 0:
            self.logger.warning(f"Found {train_nulls} null values in training data")
        
        if test_nulls > 0:
            self.logger.warning(f"Found {test_nulls} null values in test data")
        
        # Check label distribution
        label_dist = train_df['generated'].value_counts()
        self.logger.info(f"Label distribution: {label_dist.to_dict()}")
        
        self.logger.info("Data validation completed")
        return True
    
    def prepare_submission_format(self, test_df: pd.DataFrame, predictions: List[float]) -> pd.DataFrame:
        """Prepare submission file in the required format."""
        if len(predictions) != len(test_df):
            raise ValueError(f"Predictions length {len(predictions)} doesn't match test data length {len(test_df)}")
        
        submission_df = pd.DataFrame({
            'ID': test_df['ID'],
            'generated': predictions
        })
        
        return submission_df
    
    def save_submission(self, submission_df: pd.DataFrame, 
                       filename: Optional[str] = None) -> str:
        """Save submission file with versioning."""
        import os
        import datetime
        
        # Create submissions directory
        os.makedirs(self.data_config.submission_dir, exist_ok=True)
        
        if filename is None:
            # Generate timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Try to get git commit hash
            try:
                import subprocess
                result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                      capture_output=True, text=True)
                git_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
            except:
                git_hash = "unknown"
            
            filename = f"submission_{timestamp}_{git_hash}.csv"
        
        # Save versioned file
        versioned_path = os.path.join(self.data_config.submission_dir, filename)
        submission_df.to_csv(versioned_path, index=False)
        
        # Also save as latest submission for compatibility
        latest_path = self.data_config.submission_file
        submission_df.to_csv(latest_path, index=False)
        
        self.logger.info(f"Submission saved to: {versioned_path}")
        self.logger.info(f"Latest submission: {latest_path}")
        
        return versioned_path
    
    def get_data_statistics(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, any]:
        """Get comprehensive data statistics."""
        stats = {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_columns': list(train_df.columns),
            'test_columns': list(test_df.columns),
            'memory_usage': {
                'train_mb': train_df.memory_usage(deep=True).sum() / 1024 / 1024,
                'test_mb': test_df.memory_usage(deep=True).sum() / 1024 / 1024
            }
        }
        
        if 'generated' in train_df.columns:
            stats['label_distribution'] = train_df['generated'].value_counts().to_dict()
        
        return stats