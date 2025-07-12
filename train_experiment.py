#!/usr/bin/env python3
"""
Experiment-based training script with enhanced class imbalance handling.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
from typing import Dict, Any
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config, get_config_for_model, get_config_for_environment
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
import torch


def setup_logging(log_file: str = "training.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def save_training_metrics(metrics: Dict[str, Any], filename: str):
    """Save training metrics to JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def check_experiment_name_exists(experiment_name: str, base_dir: str = "models") -> bool:
    """Check if experiment name already exists."""
    experiment_dir = os.path.join(base_dir, experiment_name)
    return os.path.exists(experiment_dir)


def get_unique_experiment_name(base_name: str, base_dir: str = "models") -> str:
    """Generate unique experiment name if collision exists."""
    if not check_experiment_name_exists(base_name, base_dir):
        return base_name
    
    counter = 1
    while True:
        new_name = f"{base_name}_v{counter}"
        if not check_experiment_name_exists(new_name, base_dir):
            return new_name
        counter += 1


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train AI Text Detection Model with Experiment Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_experiment.py --name "koelectra_focal_v1" --model koelectra --loss focal
  python train_experiment.py --name "klue_bert_baseline" --model klue-bert --loss bce --cuda 0
  python train_experiment.py --name "kcbert_experiment" --model kcbert --loss focal --gpu "0,1"
  python train_experiment.py --name "cpu_test" --model klue-bert --loss focal --env cpu

Available Models:
  klue-bert     - KLUE BERT Base (klue/bert-base)
  koelectra     - KoELECTRA Base v3 (monologg/koelectra-base-v3-discriminator)
  kcbert        - KcBERT Base (beomi/kcbert-base)
  kobert        - KoBERT Base (skt/kobert-base-v1)

GPU Options:
  --cuda N      - Use single GPU N (e.g., --cuda 0)
  --gpu "N,M"   - Use multiple GPUs (e.g., --gpu "0,1" or --gpu "1,2,3")
  --env cpu     - Force CPU mode

Directory Structure:
  models/
  ├── experiment_name/
  │   ├── training.log
  │   ├── training_metrics.json
  │   ├── best_model_model_name_fold_1.pt
  │   ├── best_model_model_name_fold_2.pt
  │   └── best_model_model_name_fold_3.pt
        """
    )
    
    parser.add_argument('--name', type=str, required=True,
                       help='Experiment name (required). Will be used as subdirectory in models/')
    parser.add_argument('--model', type=str, default='klue-bert', 
                       choices=['klue-bert', 'koelectra', 'kcbert', 'kobert'],
                       help='Model to use for training (default: klue-bert)')
    parser.add_argument('--loss', type=str, default='focal',
                       choices=['focal', 'bce_weighted', 'bce'],
                       help='Loss function to use (default: focal)')
    parser.add_argument('--env', type=str, default='gpu',
                       choices=['gpu', 'cpu', 'h100', 'debug'],
                       help='Environment to use (default: gpu)')
    parser.add_argument('--gpu', type=str, default=None,
                       help='GPU IDs to use (e.g., "0", "0,1", "1,2,3"). If not specified, uses environment default')
    parser.add_argument('--cuda', type=int, default=None,
                       help='Single CUDA device ID to use (e.g., 0, 1). Overrides --gpu if specified')
    
    args = parser.parse_args()
    
    # Parse GPU configuration
    gpu_ids = None
    if args.cuda is not None:
        gpu_ids = [args.cuda]
    elif args.gpu is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu.split(',')]
    
    # Validate GPU availability
    if gpu_ids and args.env != 'cpu':
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            print("⚠️  No CUDA devices available. Switching to CPU mode.")
            args.env = 'cpu'
            gpu_ids = None
        else:
            invalid_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id >= available_gpus]
            if invalid_gpus:
                print(f"⚠️  Invalid GPU IDs: {invalid_gpus}. Available GPUs: 0-{available_gpus-1}")
                return
    
    # Generate unique experiment name
    experiment_name = get_unique_experiment_name(args.name)
    if experiment_name != args.name:
        print(f"⚠️  Experiment name '{args.name}' already exists. Using '{experiment_name}' instead.")
    
    # Create experiment directory
    experiment_dir = os.path.join("models", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Setup logging with experiment-specific log file
    log_file = os.path.join(experiment_dir, "training.log")
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    # Get model-specific config with environment and GPU settings
    config = get_config_for_model(args.model)
    env_config = get_config_for_environment(args.env, gpu_ids)
    
    # Merge configurations
    config.system = env_config.system
    config.training.loss_function = args.loss
    
    # Update model save directory to experiment directory
    config.system.model_save_dir = experiment_dir
    
    logger.info(f"🚀 EXPERIMENT-BASED TRAINING FOR AI TEXT DETECTION")
    logger.info(f"============================================================")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Model: {config.model.model_name}")
    logger.info(f"Loss Function: {config.training.loss_function}")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Device: {config.system.device}")
    if config.system.use_multi_gpu:
        logger.info(f"Multi-GPU: {config.system.gpu_ids}")
    logger.info(f"Save Directory: {experiment_dir}")
    
    try:
        # Load and preprocess data
        data_processor = DataProcessor(config.data, config.model)
        train_df, test_df = data_processor.load_data()
        train_paragraph_df = data_processor.preprocess_training_data(train_df)
        
        # Prepare training data
        train_texts = train_paragraph_df['paragraph_text'].tolist()
        train_labels = train_paragraph_df['generated'].tolist()
        
        # Create document IDs for document-aware cross-validation
        document_ids = train_paragraph_df['title'].tolist()
        
        # Initialize model trainer
        model_trainer = ModelTrainer(config)
        
        # Perform cross-validation with document-aware splits
        logger.info("Starting cross-validation with document-aware splits...")
        oof_auc, model_paths = model_trainer.cross_validate(train_texts, train_labels, document_ids)
        
        # Save training metrics
        training_metrics = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'oof_auc': oof_auc,
            'model_paths': model_paths,
            'config': config.__dict__,
            'command_args': {
                'name': args.name,
                'model': args.model,
                'loss': args.loss,
                'env': args.env
            },
            'data_stats': {
                'train_paragraphs': len(train_paragraph_df),
                'test_paragraphs': len(pd.read_csv(config.data.test_file, encoding=config.data.encoding)),
                'class_distribution': {
                    'class_0': (train_paragraph_df['generated'] == 0).mean(),
                    'class_1': (train_paragraph_df['generated'] == 1).mean()
                },
                'avg_paragraph_length': train_paragraph_df['text'].str.len().mean()
            }
        }
        
        metrics_file = os.path.join(experiment_dir, "training_metrics.json")
        save_training_metrics(training_metrics, metrics_file)
        
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"📊 Overall OOF AUC: {oof_auc:.4f}")
        logger.info(f"💾 Model checkpoints saved: {model_paths}")
        logger.info(f"📁 Experiment directory: {experiment_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()