"""
Synthetic data generation for Korean text classification.
Creates synthetic examples to improve model performance.
"""

import random
import re
from typing import List, Tuple, Dict, Optional
import logging


class SyntheticDataGenerator:
    """Synthetic data generation class."""
    
    def __init__(self, min_length: int = 50, max_length: int = 500):
        """
        Initialize synthetic data generator.
        
        Args:
            min_length: Minimum length of generated text
            max_length: Maximum length of generated text
        """
        self.min_length = min_length
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
        
        # Korean text patterns for AI-generated text
        self.ai_patterns = [
            # More formal/structured patterns
            "이러한 관점에서 볼 때,",
            "다음과 같은 측면에서 고려할 수 있다:",
            "이를 통해 우리는",
            "결론적으로 말하면,",
            "종합적으로 판단하면,",
            "이와 같은 맥락에서",
            "구체적으로 살펴보면,",
            "다양한 관점에서 분석하면,",
            
            # Repetitive patterns
            "중요한 것은",
            "필요한 것은",
            "중요한 요소는",
            "핵심적인 사항은",
            "주요한 특징은",
            
            # Structured transitions
            "첫째,", "둘째,", "셋째,",
            "먼저,", "다음으로,", "마지막으로,",
            "또한,", "그리고", "더불어",
        ]
        
        # Human-written text patterns
        self.human_patterns = [
            # More casual/natural patterns
            "그런데 말이야,",
            "사실 나는",
            "정말로 생각해보니",
            "어떻게 보면",
            "가끔 느끼는 건데",
            "내 경험으로는",
            "솔직히 말하면",
            "개인적으로는",
            
            # Emotional expressions
            "정말 놀라웠다",
            "너무 기뻤다",
            "상당히 실망했다",
            "매우 인상적이었다",
            "꽤 흥미로웠다",
            
            # Informal connectors
            "그래서 말인데",
            "그러다가",
            "그런데도",
            "어쨌든",
            "하지만 말야",
        ]
        
        # Common Korean topics
        self.topics = [
            "교육", "기술", "사회", "문화", "경제", "환경", "정치", "스포츠",
            "건강", "여행", "음식", "예술", "과학", "역사", "철학", "문학"
        ]
        
        # Common sentence structures
        self.sentence_structures = [
            "{subject}은/는 {predicate}다.",
            "{subject}이/가 {object}을/를 {action}한다.",
            "{topic}에 대해 {opinion}이다.",
            "{situation}에서 {reaction}했다.",
            "{person}은/는 {description}한 사람이다.",
        ]
    
    def generate_ai_like_text(self, topic: str, target_length: int = 200) -> str:
        """
        Generate AI-like text with formal patterns.
        
        Args:
            topic: Topic to write about
            target_length: Target length of generated text
            
        Returns:
            Generated AI-like text
        """
        text_parts = []
        current_length = 0
        
        # Start with formal introduction
        intro_pattern = random.choice([
            f"{topic}에 대해 살펴보면,",
            f"{topic}의 경우,",
            f"{topic}와 관련하여,",
            f"{topic}에 관해 분석해보면,",
        ])
        text_parts.append(intro_pattern)
        current_length += len(intro_pattern)
        
        # Add structured content
        while current_length < target_length:
            # Add AI-like pattern
            pattern = random.choice(self.ai_patterns)
            
            # Add some content
            content_templates = [
                f"이는 {topic} 분야에서 중요한 의미를 가진다.",
                f"다양한 관점에서 {topic}을 고려할 필요가 있다.",
                f"{topic}의 발전을 위해서는 체계적인 접근이 필요하다.",
                f"현재 {topic}의 동향을 살펴보면 여러 가지 특징이 나타난다.",
                f"{topic}에 대한 이해를 높이기 위해 다음과 같은 점을 고려해야 한다.",
            ]
            
            content = random.choice(content_templates)
            
            sentence = f" {pattern} {content}"
            text_parts.append(sentence)
            current_length += len(sentence)
            
            if current_length >= target_length:
                break
        
        # Add formal conclusion
        conclusion_pattern = random.choice([
            f"결론적으로, {topic}은 지속적인 관심과 연구가 필요한 분야이다.",
            f"이상으로 {topic}에 대한 분석을 마친다.",
            f"향후 {topic}의 발전을 위해 더 많은 연구가 필요할 것으로 보인다.",
        ])
        text_parts.append(f" {conclusion_pattern}")
        
        return "".join(text_parts)
    
    def generate_human_like_text(self, topic: str, target_length: int = 200) -> str:
        """
        Generate human-like text with natural patterns.
        
        Args:
            topic: Topic to write about
            target_length: Target length of generated text
            
        Returns:
            Generated human-like text
        """
        text_parts = []
        current_length = 0
        
        # Start with personal/casual introduction
        intro_pattern = random.choice([
            f"내가 {topic}에 대해 생각해보니,",
            f"{topic} 얘기를 하자면,",
            f"요즘 {topic}에 관심이 많은데,",
            f"{topic}에 대해 느끼는 점이 있다.",
        ])
        text_parts.append(intro_pattern)
        current_length += len(intro_pattern)
        
        # Add natural content
        while current_length < target_length:
            # Add human-like pattern
            pattern = random.choice(self.human_patterns)
            
            # Add personal content
            content_templates = [
                f"{topic}은 정말 흥미로운 분야야.",
                f"가끔 {topic}에 대해 생각해보곤 한다.",
                f"{topic}을 접하면서 많은 것을 느꼈다.",
                f"내 주변에서도 {topic}에 대한 관심이 높아지고 있다.",
                f"{topic}의 변화를 보면서 시대의 흐름을 느낀다.",
            ]
            
            content = random.choice(content_templates)
            
            sentence = f" {pattern} {content}"
            text_parts.append(sentence)
            current_length += len(sentence)
            
            if current_length >= target_length:
                break
        
        # Add personal conclusion
        conclusion_pattern = random.choice([
            f"앞으로도 {topic}에 대해 계속 관심을 가져야겠다.",
            f"{topic}은 우리 삶에 중요한 영향을 미치는 것 같다.",
            f"이런 점에서 {topic}의 가치를 새롭게 인식하게 된다.",
        ])
        text_parts.append(f" {conclusion_pattern}")
        
        return "".join(text_parts)
    
    def generate_synthetic_data(self, num_samples: int = 1000, 
                              balance_ratio: float = 0.5) -> Tuple[List[str], List[int]]:
        """
        Generate synthetic training data.
        
        Args:
            num_samples: Number of samples to generate
            balance_ratio: Ratio of AI-generated samples
            
        Returns:
            Tuple of (texts, labels)
        """
        texts = []
        labels = []
        
        num_ai_samples = int(num_samples * balance_ratio)
        num_human_samples = num_samples - num_ai_samples
        
        self.logger.info(f"Generating {num_ai_samples} AI-like samples and {num_human_samples} human-like samples")
        
        # Generate AI-like samples
        for i in range(num_ai_samples):
            if i % 100 == 0:
                self.logger.info(f"Generated {i}/{num_ai_samples} AI samples")
            
            topic = random.choice(self.topics)
            target_length = random.randint(self.min_length, self.max_length)
            
            text = self.generate_ai_like_text(topic, target_length)
            texts.append(text)
            labels.append(1)  # AI-generated
        
        # Generate human-like samples
        for i in range(num_human_samples):
            if i % 100 == 0:
                self.logger.info(f"Generated {i}/{num_human_samples} human samples")
            
            topic = random.choice(self.topics)
            target_length = random.randint(self.min_length, self.max_length)
            
            text = self.generate_human_like_text(topic, target_length)
            texts.append(text)
            labels.append(0)  # Human-written
        
        # Shuffle the data
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        
        self.logger.info(f"Generated {len(texts)} synthetic samples")
        
        return list(texts), list(labels)
    
    def enhance_existing_data(self, texts: List[str], labels: List[int],
                            target_ratio: float = 0.3) -> Tuple[List[str], List[int]]:
        """
        Enhance existing data with synthetic samples.
        
        Args:
            texts: Existing texts
            labels: Existing labels
            target_ratio: Target ratio for minority class
            
        Returns:
            Tuple of (enhanced_texts, enhanced_labels)
        """
        # Count current classes
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        current_pos_ratio = pos_count / len(labels)
        
        self.logger.info(f"Current positive ratio: {current_pos_ratio:.3f}")
        self.logger.info(f"Target positive ratio: {target_ratio:.3f}")
        
        # If already balanced, return original
        if current_pos_ratio >= target_ratio:
            return texts, labels
        
        # Calculate needed samples
        needed_pos = int(target_ratio * len(labels) / (1 - target_ratio)) - pos_count
        
        if needed_pos <= 0:
            return texts, labels
        
        self.logger.info(f"Generating {needed_pos} synthetic positive samples")
        
        # Generate synthetic AI-like samples
        synthetic_texts = []
        synthetic_labels = []
        
        for _ in range(needed_pos):
            topic = random.choice(self.topics)
            target_length = random.randint(self.min_length, self.max_length)
            
            text = self.generate_ai_like_text(topic, target_length)
            synthetic_texts.append(text)
            synthetic_labels.append(1)
        
        # Combine with existing data
        enhanced_texts = texts + synthetic_texts
        enhanced_labels = labels + synthetic_labels
        
        # Shuffle
        combined = list(zip(enhanced_texts, enhanced_labels))
        random.shuffle(combined)
        enhanced_texts, enhanced_labels = zip(*combined)
        
        final_pos_ratio = sum(enhanced_labels) / len(enhanced_labels)
        self.logger.info(f"Final dataset size: {len(enhanced_texts)}")
        self.logger.info(f"Final positive ratio: {final_pos_ratio:.3f}")
        
        return list(enhanced_texts), list(enhanced_labels)