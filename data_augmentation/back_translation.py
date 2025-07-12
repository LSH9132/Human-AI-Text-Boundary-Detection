"""
Back-translation augmentation for Korean text.
Uses translation models to create paraphrased versions of text.
"""

import random
from typing import List, Optional, Tuple
import logging


class BackTranslator:
    """Back-translation augmentation class."""
    
    def __init__(self, intermediate_languages: List[str] = None):
        """
        Initialize back translator.
        
        Args:
            intermediate_languages: List of intermediate languages for back-translation
        """
        self.logger = logging.getLogger(__name__)
        
        if intermediate_languages is None:
            intermediate_languages = ['en', 'ja', 'zh']
        
        self.intermediate_languages = intermediate_languages
        self.translator = None
        
        # Try to initialize translator
        try:
            from googletrans import Translator
            self.translator = Translator()
            self.logger.info("Google Translate initialized successfully")
        except ImportError:
            self.logger.warning("googletrans not available. Back-translation disabled.")
        except Exception as e:
            self.logger.warning(f"Failed to initialize translator: {e}")
    
    def is_available(self) -> bool:
        """Check if back-translation is available."""
        return self.translator is not None
    
    def back_translate(self, text: str, intermediate_lang: str = 'en') -> Optional[str]:
        """
        Perform back-translation through an intermediate language.
        
        Args:
            text: Input Korean text
            intermediate_lang: Intermediate language code
            
        Returns:
            Back-translated text or None if failed
        """
        if not self.is_available():
            return None
        
        try:
            # Translate to intermediate language
            intermediate = self.translator.translate(text, src='ko', dest=intermediate_lang)
            
            if intermediate is None or intermediate.text is None:
                return None
            
            # Translate back to Korean
            back_translated = self.translator.translate(intermediate.text, src=intermediate_lang, dest='ko')
            
            if back_translated is None or back_translated.text is None:
                return None
            
            return back_translated.text
            
        except Exception as e:
            self.logger.debug(f"Back-translation failed: {e}")
            return None
    
    def augment_text(self, text: str, num_augmented: int = 1) -> List[str]:
        """
        Generate back-translated versions of the text.
        
        Args:
            text: Input Korean text
            num_augmented: Number of augmented versions to generate
            
        Returns:
            List of back-translated texts
        """
        if not self.is_available():
            self.logger.warning("Back-translation not available")
            return []
        
        augmented_texts = []
        
        for _ in range(num_augmented):
            # Randomly select intermediate language
            intermediate_lang = random.choice(self.intermediate_languages)
            
            # Perform back-translation
            back_translated = self.back_translate(text, intermediate_lang)
            
            if back_translated and back_translated != text:
                augmented_texts.append(back_translated)
        
        return augmented_texts
    
    def augment_batch(self, texts: List[str], labels: List[int],
                     target_ratio: float = 0.3, max_augmented_per_text: int = 2) -> Tuple[List[str], List[int]]:
        """
        Augment batch of texts with back-translation.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            target_ratio: Target ratio for minority class
            max_augmented_per_text: Maximum augmented versions per text
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        if not self.is_available():
            self.logger.warning("Back-translation not available, returning original data")
            return texts, labels
        
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
        
        self.logger.info(f"Generating {needed_pos} positive samples via back-translation")
        
        # Get positive samples
        pos_texts = [texts[i] for i in range(len(texts)) if labels[i] == 1]
        
        # Generate augmented positive samples
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        generated_count = 0
        attempts = 0
        max_attempts = needed_pos * 3  # Limit attempts to avoid infinite loop
        
        while generated_count < needed_pos and attempts < max_attempts:
            # Randomly select a positive sample
            source_text = random.choice(pos_texts)
            
            # Generate augmented versions
            aug_versions = self.augment_text(source_text, num_augmented=1)
            
            if aug_versions:
                augmented_texts.append(aug_versions[0])
                augmented_labels.append(1)
                generated_count += 1
            
            attempts += 1
        
        if generated_count < needed_pos:
            self.logger.warning(f"Only generated {generated_count} out of {needed_pos} samples")
        
        self.logger.info(f"Final dataset size: {len(augmented_texts)}")
        final_pos_ratio = sum(augmented_labels) / len(augmented_labels)
        self.logger.info(f"Final positive ratio: {final_pos_ratio:.3f}")
        
        return augmented_texts, augmented_labels


class SimpleParaphraser:
    """Simple paraphrasing without external translation services."""
    
    def __init__(self):
        """Initialize simple paraphraser."""
        self.logger = logging.getLogger(__name__)
        
        # Simple paraphrasing patterns
        self.patterns = [
            # Sentence structure changes
            (r'(.+)는 (.+)이다', r'\1은 \2다'),
            (r'(.+)은 (.+)이다', r'\1는 \2다'),
            (r'(.+)가 (.+)이다', r'\1이 \2다'),
            (r'(.+)이 (.+)이다', r'\1가 \2다'),
            
            # Connector changes
            (r'그리고', r'또한'),
            (r'하지만', r'그러나'),
            (r'그래서', r'따라서'),
            (r'때문에', r'로 인해'),
            
            # Tense changes
            (r'한다', r'하고 있다'),
            (r'이다', r'이라고 할 수 있다'),
        ]
    
    def paraphrase(self, text: str) -> str:
        """
        Simple paraphrasing using pattern replacement.
        
        Args:
            text: Input text
            
        Returns:
            Paraphrased text
        """
        import re
        
        paraphrased = text
        
        # Apply random subset of patterns
        selected_patterns = random.sample(self.patterns, min(3, len(self.patterns)))
        
        for pattern, replacement in selected_patterns:
            paraphrased = re.sub(pattern, replacement, paraphrased)
        
        return paraphrased
    
    def augment_text(self, text: str, num_augmented: int = 1) -> List[str]:
        """
        Generate paraphrased versions of the text.
        
        Args:
            text: Input text
            num_augmented: Number of augmented versions to generate
            
        Returns:
            List of paraphrased texts
        """
        augmented_texts = []
        
        for _ in range(num_augmented):
            paraphrased = self.paraphrase(text)
            
            if paraphrased != text and paraphrased not in augmented_texts:
                augmented_texts.append(paraphrased)
        
        return augmented_texts