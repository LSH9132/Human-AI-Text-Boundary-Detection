"""
Balanced sampling techniques for addressing class imbalance.
Provides various sampling strategies to balance training data.
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
import logging


class BalancedSampler:
    """Balanced sampling class for handling class imbalance."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize balanced sampler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        random.seed(random_state)
        np.random.seed(random_state)
    
    def oversample_minority(self, texts: List[str], labels: List[int],
                          target_ratio: float = 0.3) -> Tuple[List[str], List[int]]:
        """
        Oversample minority class to achieve target ratio.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            target_ratio: Target ratio for minority class
            
        Returns:
            Tuple of (oversampled_texts, oversampled_labels)
        """
        # Count classes
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        total_count = len(labels)
        
        current_pos_ratio = pos_count / total_count
        
        self.logger.info(f"Current class distribution:")
        self.logger.info(f"  Positive: {pos_count} ({current_pos_ratio:.3f})")
        self.logger.info(f"  Negative: {neg_count} ({1-current_pos_ratio:.3f})")
        self.logger.info(f"Target positive ratio: {target_ratio:.3f}")
        
        # If already balanced enough, return original
        if current_pos_ratio >= target_ratio:
            self.logger.info("Data already balanced enough")
            return texts, labels
        
        # Calculate how many positive samples to generate
        needed_total = int(neg_count / (1 - target_ratio))
        needed_pos = needed_total - total_count
        
        if needed_pos <= 0:
            return texts, labels
        
        self.logger.info(f"Oversampling {needed_pos} positive samples")
        
        # Get positive samples
        pos_indices = [i for i in range(len(texts)) if labels[i] == 1]
        
        # Generate oversampled data
        oversampled_texts = texts.copy()
        oversampled_labels = labels.copy()
        
        for _ in range(needed_pos):
            # Randomly select a positive sample
            idx = random.choice(pos_indices)
            oversampled_texts.append(texts[idx])
            oversampled_labels.append(1)
        
        # Shuffle the data
        combined = list(zip(oversampled_texts, oversampled_labels))
        random.shuffle(combined)
        oversampled_texts, oversampled_labels = zip(*combined)
        
        final_pos_ratio = sum(oversampled_labels) / len(oversampled_labels)
        self.logger.info(f"Final dataset size: {len(oversampled_texts)}")
        self.logger.info(f"Final positive ratio: {final_pos_ratio:.3f}")
        
        return list(oversampled_texts), list(oversampled_labels)
    
    def undersample_majority(self, texts: List[str], labels: List[int],
                           target_ratio: float = 0.3) -> Tuple[List[str], List[int]]:
        """
        Undersample majority class to achieve target ratio.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            target_ratio: Target ratio for minority class
            
        Returns:
            Tuple of (undersampled_texts, undersampled_labels)
        """
        # Count classes
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        
        current_pos_ratio = pos_count / len(labels)
        
        self.logger.info(f"Current positive ratio: {current_pos_ratio:.3f}")
        self.logger.info(f"Target positive ratio: {target_ratio:.3f}")
        
        # If already balanced enough, return original
        if current_pos_ratio >= target_ratio:
            self.logger.info("Data already balanced enough")
            return texts, labels
        
        # Calculate how many negative samples to keep
        target_neg_count = int(pos_count * (1 - target_ratio) / target_ratio)
        
        if target_neg_count >= neg_count:
            self.logger.info("No undersampling needed")
            return texts, labels
        
        self.logger.info(f"Undersampling negative class from {neg_count} to {target_neg_count}")
        
        # Get indices
        pos_indices = [i for i in range(len(texts)) if labels[i] == 1]
        neg_indices = [i for i in range(len(texts)) if labels[i] == 0]
        
        # Randomly select negative samples
        selected_neg_indices = random.sample(neg_indices, target_neg_count)
        
        # Combine selected indices
        selected_indices = pos_indices + selected_neg_indices
        random.shuffle(selected_indices)
        
        # Create undersampled data
        undersampled_texts = [texts[i] for i in selected_indices]
        undersampled_labels = [labels[i] for i in selected_indices]
        
        final_pos_ratio = sum(undersampled_labels) / len(undersampled_labels)
        self.logger.info(f"Final dataset size: {len(undersampled_texts)}")
        self.logger.info(f"Final positive ratio: {final_pos_ratio:.3f}")
        
        return undersampled_texts, undersampled_labels
    
    def combined_sampling(self, texts: List[str], labels: List[int],
                         target_ratio: float = 0.3, 
                         oversample_ratio: float = 0.7) -> Tuple[List[str], List[int]]:
        """
        Combine oversampling and undersampling.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            target_ratio: Target ratio for minority class
            oversample_ratio: Ratio of oversampling vs undersampling
            
        Returns:
            Tuple of (resampled_texts, resampled_labels)
        """
        # Count classes
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        
        current_pos_ratio = pos_count / len(labels)
        
        self.logger.info(f"Combined sampling approach")
        self.logger.info(f"Current positive ratio: {current_pos_ratio:.3f}")
        self.logger.info(f"Target positive ratio: {target_ratio:.3f}")
        
        if current_pos_ratio >= target_ratio:
            return texts, labels
        
        # Calculate target counts
        total_target = int(pos_count / target_ratio)
        neg_target = total_target - pos_count
        
        # Decide how much to oversample vs undersample
        oversample_amount = int(oversample_ratio * (neg_count - neg_target))
        undersample_amount = neg_count - neg_target - oversample_amount
        
        self.logger.info(f"Will oversample {oversample_amount} positive samples")
        self.logger.info(f"Will undersample {undersample_amount} negative samples")
        
        # Get indices
        pos_indices = [i for i in range(len(texts)) if labels[i] == 1]
        neg_indices = [i for i in range(len(texts)) if labels[i] == 0]
        
        # Undersample negative class
        selected_neg_indices = random.sample(neg_indices, neg_target)
        
        # Create base dataset
        selected_indices = pos_indices + selected_neg_indices
        resampled_texts = [texts[i] for i in selected_indices]
        resampled_labels = [labels[i] for i in selected_indices]
        
        # Oversample positive class
        for _ in range(oversample_amount):
            idx = random.choice(pos_indices)
            resampled_texts.append(texts[idx])
            resampled_labels.append(1)
        
        # Shuffle
        combined = list(zip(resampled_texts, resampled_labels))
        random.shuffle(combined)
        resampled_texts, resampled_labels = zip(*combined)
        
        final_pos_ratio = sum(resampled_labels) / len(resampled_labels)
        self.logger.info(f"Final dataset size: {len(resampled_texts)}")
        self.logger.info(f"Final positive ratio: {final_pos_ratio:.3f}")
        
        return list(resampled_texts), list(resampled_labels)
    
    def stratified_sampling(self, texts: List[str], labels: List[int],
                          sample_size: int, target_ratio: float = 0.3) -> Tuple[List[str], List[int]]:
        """
        Create stratified sample with target ratio.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            sample_size: Target sample size
            target_ratio: Target ratio for minority class
            
        Returns:
            Tuple of (sampled_texts, sampled_labels)
        """
        # Calculate target counts
        pos_target = int(sample_size * target_ratio)
        neg_target = sample_size - pos_target
        
        # Get indices
        pos_indices = [i for i in range(len(texts)) if labels[i] == 1]
        neg_indices = [i for i in range(len(texts)) if labels[i] == 0]
        
        # Sample with replacement if needed
        if pos_target > len(pos_indices):
            selected_pos_indices = random.choices(pos_indices, k=pos_target)
        else:
            selected_pos_indices = random.sample(pos_indices, pos_target)
        
        if neg_target > len(neg_indices):
            selected_neg_indices = random.choices(neg_indices, k=neg_target)
        else:
            selected_neg_indices = random.sample(neg_indices, neg_target)
        
        # Combine and shuffle
        selected_indices = selected_pos_indices + selected_neg_indices
        random.shuffle(selected_indices)
        
        sampled_texts = [texts[i] for i in selected_indices]
        sampled_labels = [labels[i] for i in selected_indices]
        
        final_pos_ratio = sum(sampled_labels) / len(sampled_labels)
        self.logger.info(f"Stratified sample size: {len(sampled_texts)}")
        self.logger.info(f"Final positive ratio: {final_pos_ratio:.3f}")
        
        return sampled_texts, sampled_labels
    
    def get_class_weights(self, labels: List[int]) -> Dict[int, float]:
        """
        Calculate class weights for loss function.
        
        Args:
            labels: List of labels
            
        Returns:
            Dictionary mapping class to weight
        """
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        # Calculate weights inversely proportional to class frequencies
        weights = {}
        for class_label, count in class_counts.items():
            weights[class_label] = total_samples / (len(class_counts) * count)
        
        self.logger.info(f"Class weights: {weights}")
        
        return weights