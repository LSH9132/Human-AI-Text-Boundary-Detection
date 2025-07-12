"""
Korean text augmentation techniques.
Provides various methods to augment Korean text while preserving meaning.
"""

import re
import random
from typing import List, Dict, Tuple, Optional
import logging


class KoreanTextAugmenter:
    """Korean text augmentation class."""
    
    def __init__(self, aug_prob: float = 0.3):
        """
        Initialize Korean text augmenter.
        
        Args:
            aug_prob: Probability of applying augmentation to each sentence
        """
        self.aug_prob = aug_prob
        self.logger = logging.getLogger(__name__)
        
        # Korean specific patterns
        self.korean_pattern = re.compile(r'[가-힣]+')
        self.punctuation_pattern = re.compile(r'[.,!?;:]')
        
        # Common Korean synonyms and variations
        self.synonyms = {
            '그리고': ['또한', '그래서', '더불어', '아울러'],
            '하지만': ['그러나', '그런데', '다만', '단지'],
            '때문에': ['로 인해', '의해서', '으로써'],
            '중요한': ['주요한', '핵심적인', '필수적인', '중대한'],
            '문제': ['이슈', '과제', '사안', '쟁점'],
            '방법': ['방식', '수단', '기법', '접근법'],
            '생각': ['의견', '견해', '관점', '판단'],
            '사람': ['인간', '개인', '사람들', '이들'],
            '지금': ['현재', '요즘', '오늘날', '최근'],
            '많은': ['다수의', '여러', '수많은', '다양한']
        }
        
        # Sentence connectors
        self.connectors = {
            '원인': ['때문에', '로 인해', '의해서', '으로써'],
            '결과': ['따라서', '그러므로', '결국', '그래서'],
            '대조': ['하지만', '그러나', '반면에', '그런데'],
            '추가': ['그리고', '또한', '더불어', '아울러'],
            '강조': ['특히', '무엇보다', '중요한 것은', '더욱이']
        }
    
    def augment_text(self, text: str, num_augmented: int = 1) -> List[str]:
        """
        Generate augmented versions of the input text.
        
        Args:
            text: Input Korean text
            num_augmented: Number of augmented versions to generate
            
        Returns:
            List of augmented texts
        """
        augmented_texts = []
        
        for _ in range(num_augmented):
            # Apply multiple augmentation techniques
            augmented = text
            
            # Apply augmentations with probability
            if random.random() < self.aug_prob:
                augmented = self.synonym_replacement(augmented)
            
            if random.random() < self.aug_prob:
                augmented = self.sentence_reordering(augmented)
                
            if random.random() < self.aug_prob:
                augmented = self.connector_variation(augmented)
                
            if random.random() < self.aug_prob:
                augmented = self.punctuation_variation(augmented)
            
            # Only add if different from original
            if augmented != text and augmented not in augmented_texts:
                augmented_texts.append(augmented)
        
        return augmented_texts
    
    def synonym_replacement(self, text: str) -> str:
        """Replace words with Korean synonyms."""
        for word, synonyms in self.synonyms.items():
            if word in text:
                synonym = random.choice(synonyms)
                text = text.replace(word, synonym, 1)  # Replace only first occurrence
        
        return text
    
    def sentence_reordering(self, text: str) -> str:
        """Reorder sentences in the text."""
        sentences = self._split_korean_sentences(text)
        
        if len(sentences) <= 1:
            return text
        
        # Randomly reorder sentences (but keep first/last in reasonable positions)
        if len(sentences) > 3:
            middle_sentences = sentences[1:-1]
            random.shuffle(middle_sentences)
            reordered = [sentences[0]] + middle_sentences + [sentences[-1]]
        else:
            reordered = sentences.copy()
            random.shuffle(reordered)
        
        return ' '.join(reordered)
    
    def connector_variation(self, text: str) -> str:
        """Vary sentence connectors."""
        for connector_type, connectors in self.connectors.items():
            for connector in connectors:
                if connector in text:
                    new_connector = random.choice(connectors)
                    text = text.replace(connector, new_connector, 1)
                    break
        
        return text
    
    def punctuation_variation(self, text: str) -> str:
        """Slightly vary punctuation."""
        # Add or remove some punctuation
        if random.random() < 0.3:
            # Add emphasis
            text = text.replace('.', '.')
            text = text.replace('!', '!')
        
        return text
    
    def _split_korean_sentences(self, text: str) -> List[str]:
        """Split Korean text into sentences."""
        # Simple sentence splitting for Korean
        sentences = re.split(r'[.!?]\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def word_dropout(self, text: str, dropout_prob: float = 0.1) -> str:
        """Randomly drop some words (careful with Korean)."""
        words = text.split()
        
        if len(words) <= 3:  # Don't drop from very short texts
            return text
        
        # Keep important words (nouns, verbs) and drop function words
        filtered_words = []
        for word in words:
            if random.random() > dropout_prob or len(word) <= 1:
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def add_noise(self, text: str, noise_prob: float = 0.05) -> str:
        """Add slight noise to text (typos, spacing)."""
        chars = list(text)
        
        for i in range(len(chars)):
            if random.random() < noise_prob:
                # Add random spacing or remove spacing
                if chars[i] == ' ':
                    chars[i] = ''
                elif i < len(chars) - 1 and chars[i+1] != ' ':
                    chars[i] = chars[i] + ' '
        
        return ''.join(chars)
    
    def augment_for_balance(self, texts: List[str], labels: List[int], 
                          target_ratio: float = 0.3) -> Tuple[List[str], List[int]]:
        """
        Augment minority class to achieve better balance.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            target_ratio: Target ratio for minority class
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        # Count classes
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        total_count = len(labels)
        
        current_pos_ratio = pos_count / total_count
        
        self.logger.info(f"Current positive ratio: {current_pos_ratio:.3f}")
        self.logger.info(f"Target positive ratio: {target_ratio:.3f}")
        
        # If already balanced enough, return original
        if current_pos_ratio >= target_ratio:
            return texts, labels
        
        # Calculate how many positive samples to generate
        needed_pos = int(target_ratio * total_count / (1 - target_ratio)) - pos_count
        
        if needed_pos <= 0:
            return texts, labels
        
        self.logger.info(f"Generating {needed_pos} positive samples via augmentation")
        
        # Get positive samples
        pos_texts = [texts[i] for i in range(len(texts)) if labels[i] == 1]
        
        # Generate augmented positive samples
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        for _ in range(needed_pos):
            # Randomly select a positive sample
            source_text = random.choice(pos_texts)
            
            # Generate augmented version
            aug_versions = self.augment_text(source_text, num_augmented=1)
            
            if aug_versions:
                augmented_texts.append(aug_versions[0])
                augmented_labels.append(1)
        
        self.logger.info(f"Final dataset size: {len(augmented_texts)}")
        final_pos_ratio = sum(augmented_labels) / len(augmented_labels)
        self.logger.info(f"Final positive ratio: {final_pos_ratio:.3f}")
        
        return augmented_texts, augmented_labels