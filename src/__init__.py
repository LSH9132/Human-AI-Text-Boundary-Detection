"""
AI Text Detection Project - Source Module
"""

from .config import Config, get_config, get_config_for_environment
from .data_processor import DataProcessor, TextDataset
from .model_trainer import ModelTrainer
from .predictor import Predictor
from .utils import ProjectManager, setup_project_management

__version__ = "1.0.0"
__all__ = [
    "Config",
    "get_config", 
    "get_config_for_environment",
    "DataProcessor",
    "TextDataset", 
    "ModelTrainer",
    "Predictor",
    "ProjectManager",
    "setup_project_management"
]