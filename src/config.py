"""
Configuration file for AI Text Detection project.
Centralized configuration management for all model and training parameters.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_name: str = 'klue/bert-base'
    max_length: int = 512
    num_labels: int = 1
    dropout_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 3
    n_splits: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 3
    
    # Context adjustment parameters
    context_weight: float = 0.3  # 30% document average, 70% individual prediction


@dataclass
class DataConfig:
    """Data configuration parameters."""
    train_file: str = 'data/train.csv'
    test_file: str = 'data/test.csv'
    sample_submission_file: str = 'data/sample_submission.csv'
    submission_file: str = 'submission.csv'
    encoding: str = 'utf-8-sig'
    
    # Data processing parameters
    min_paragraph_length: int = 10
    max_paragraphs_per_document: int = 50


@dataclass
class SystemConfig:
    """System configuration parameters."""
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    
    # Model saving
    model_save_dir: str = 'models'
    checkpoint_pattern: str = 'best_model_fold_{fold}.pt'
    
    # Logging
    log_level: str = 'INFO'
    log_file: str = 'training.log'
    wandb_project: str = 'ai-text-detection'
    wandb_enabled: bool = False


@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    system: SystemConfig = SystemConfig()
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure model save directory exists
        os.makedirs(self.system.model_save_dir, exist_ok=True)
        
        # Auto-detect device if set to 'auto'
        if self.system.device == 'auto':
            import torch
            self.system.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'system': self.system.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            system=SystemConfig(**config_dict.get('system', {}))
        )


def get_config() -> Config:
    """Get the default configuration."""
    return Config()


def get_config_for_environment(env: str = 'default') -> Config:
    """Get configuration for specific environment."""
    config = Config()
    
    if env == 'gpu':
        config.system.device = 'cuda'
        config.training.batch_size = 16
        config.system.mixed_precision = True
    elif env == 'cpu':
        config.system.device = 'cpu'
        config.training.batch_size = 4
        config.system.mixed_precision = False
    elif env == 'debug':
        config.training.epochs = 1
        config.training.n_splits = 2
        config.system.log_level = 'DEBUG'
    
    return config