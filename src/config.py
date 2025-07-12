"""
Configuration file for AI Text Detection project.
Centralized configuration management for all model and training parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_name: str = 'klue/bert-base'
    max_length: int = 256  # Reduced from 512 to 256 for faster training
    num_labels: int = 1
    dropout_rate: float = 0.1
    
    # Alternative Korean models for experimentation
    alternative_models: Dict[str, str] = None
    
    def __post_init__(self):
        if self.alternative_models is None:
            self.alternative_models = {
                'klue-bert': 'klue/bert-base',
                'koelectra': 'monologg/koelectra-base-v3-discriminator',
                'kcbert': 'beomi/kcbert-base',
                'kobert': 'skt/kobert-base-v1',
                'kobigbird': 'monologg/kobigbird-bert-base'
            }


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 32  # Reduced from 64 to 32 for stability
    learning_rate: float = 2e-5
    epochs: int = 3
    n_splits: int = 3  # Reduced from 5 to 3 for faster training
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 2  # Reduced from 3 to 2 for faster convergence
    
    # Optimization parameters  
    gradient_accumulation_steps: int = 1  # No accumulation needed with larger batch size
    
    # Context adjustment parameters
    context_weight: float = 0.3  # 30% document average, 70% individual prediction
    
    # Loss function selection
    loss_function: str = 'focal'  # 'bce_weighted', 'focal', 'bce'


@dataclass
class DataConfig:
    """Data configuration parameters."""
    train_file: str = 'data/train.csv'
    test_file: str = 'data/test.csv'
    sample_submission_file: str = 'data/sample_submission.csv'
    submission_file: str = 'submission.csv'
    submission_dir: str = 'submissions'
    encoding: str = 'utf-8-sig'
    
    # Data processing parameters - OPTIMIZED for better representation
    min_paragraph_length: int = 20  # Increased to filter out very short paragraphs
    max_paragraphs_per_document: int = 10  # Increased from 3 to 10 for better document representation


@dataclass
class SystemConfig:
    """System configuration parameters."""
    device: str = 'cuda:1'  # Use GPU 1 instead of auto
    num_workers: int = 8  # Increased from 4 to 8 for faster data loading
    pin_memory: bool = True
    mixed_precision: bool = True
    
    # Multi-GPU settings
    use_multi_gpu: bool = False
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    # Model saving
    model_save_dir: str = 'models'
    checkpoint_pattern: str = 'best_model_{model_name}_fold_{fold}.pt'
    
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


def get_config_for_environment(env: str = 'default', gpu_ids: List[int] = None) -> Config:
    """Get configuration for specific environment."""
    config = Config()
    
    if env == 'gpu':
        if gpu_ids:
            config.system.device = f'cuda:{gpu_ids[0]}'
            config.system.gpu_ids = gpu_ids
            config.system.use_multi_gpu = len(gpu_ids) > 1
        else:
            config.system.device = 'cuda:1'  # Force GPU 1 for multi-GPU optimization
        config.training.batch_size = 32   # Increased from 16 for speed
        config.system.mixed_precision = True
    elif env == 'h100':
        # H100 optimization for vast.ai deployment
        if gpu_ids:
            config.system.device = f'cuda:{gpu_ids[0]}'
            config.system.gpu_ids = gpu_ids
            config.system.use_multi_gpu = len(gpu_ids) > 1
        else:
            config.system.device = 'cuda'
        config.training.batch_size = 256  # Utilize H100's 80GB memory
        config.model.max_length = 512     # Restore full length for better performance
        config.system.mixed_precision = True
        config.system.num_workers = 16   # High-performance data loading
        config.training.gradient_accumulation_steps = 1
        config.data.max_paragraphs_per_document = 5  # Slightly more data
        config.system.log_level = 'INFO'
    elif env == 'cpu':
        config.system.device = 'cpu'
        config.training.batch_size = 4
        config.system.mixed_precision = False
    elif env == 'debug':
        config.training.epochs = 1
        config.training.n_splits = 2
        config.system.log_level = 'DEBUG'
    elif env == 'koelectra':
        # KoELECTRA optimized configuration
        config.model.model_name = 'monologg/koelectra-base-v3-discriminator'
        config.model.max_length = 512      # KoELECTRA works better with longer sequences
        config.training.batch_size = 16    # Smaller batch for stability
        config.training.learning_rate = 3e-5  # Slightly higher learning rate
        config.training.loss_function = 'focal'
        config.data.max_paragraphs_per_document = 15  # More paragraphs for better context
        if gpu_ids:
            config.system.device = f'cuda:{gpu_ids[0]}'
            config.system.gpu_ids = gpu_ids
            config.system.use_multi_gpu = len(gpu_ids) > 1
        else:
            config.system.device = 'cuda:1'
        config.system.mixed_precision = True
    
    return config


def get_config_for_model(model_name: str = 'klue-bert') -> Config:
    """Get configuration optimized for specific model."""
    config = Config()
    
    if model_name == 'koelectra':
        config.model.model_name = 'monologg/koelectra-base-v3-discriminator'
        config.model.max_length = 512
        config.training.batch_size = 16
        config.training.learning_rate = 3e-5
        config.training.loss_function = 'focal'
        config.data.max_paragraphs_per_document = 15
    elif model_name == 'kcbert':
        config.model.model_name = 'beomi/kcbert-base'
        config.model.max_length = 300
        config.training.batch_size = 24
        config.training.learning_rate = 2e-5
        config.training.loss_function = 'focal'
        config.data.max_paragraphs_per_document = 12
    elif model_name == 'kobert':
        config.model.model_name = 'skt/kobert-base-v1'
        config.model.max_length = 256
        config.training.batch_size = 32
        config.training.learning_rate = 2e-5
        config.training.loss_function = 'focal'
        config.data.max_paragraphs_per_document = 10
    else:  # klue-bert (default)
        config.model.model_name = 'klue/bert-base'
        config.model.max_length = 256
        config.training.batch_size = 32
        config.training.learning_rate = 2e-5
        config.training.loss_function = 'focal'
        config.data.max_paragraphs_per_document = 10
    
    return config