# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning competition project for AI-generated text detection using Korean BERT. The project implements a binary classification model to detect whether text paragraphs are AI-generated or human-written.

## Commands

### Running the Model

#### Quick Validation
```bash
# Test environment and basic functionality
.venv/Scripts/python.exe simple_test.py

# Validate data structure
.venv/Scripts/python.exe test_data.py

# Quick model test with sample data
.venv/Scripts/python.exe test_model.py
```

#### Full Training
```bash
# For CPU environment
.venv/Scripts/python.exe main.py

# For CUDA environment (when available)
.venv/Scripts/python.exe main_cuda.py
```

This will:
- Load and preprocess training/test data from the `data/` directory (1.2M+ samples)
- Train a KLUE/BERT-base model using 5-fold cross-validation
- Apply context-aware prediction with document-level consistency
- Generate predictions and save them to `submission.csv`

### Dependencies
Install required packages:
```bash
# Option 1: Using pip
pip install -r requirements.txt

# Option 2: Using install script
.venv/Scripts/python.exe install_packages.py
```

Key packages:
- pandas, numpy, scikit-learn
- torch, transformers
- Korean BERT model: `klue/bert-base`

### Environment Setup
```bash
# Activate virtual environment
source .venv/Scripts/activate  # Windows
source .venv/bin/activate      # Linux/Mac
```

## Architecture

### Core Components

1. **Data Processing** (`load_and_preprocess_data()` at main.py:26)
   - Splits full text into paragraphs for paragraph-level classification
   - Creates expanded training dataset from original samples
   - Handles Korean text preprocessing

2. **Model Architecture** 
   - Uses `AutoModelForSequenceClassification` with KLUE/BERT-base
   - Binary classification (1 output label)
   - Custom `TextDataset` class for PyTorch data loading

3. **Training Strategy**
   - 5-fold stratified cross-validation
   - AdamW optimizer with 2e-5 learning rate
   - BCEWithLogitsLoss for binary classification
   - Early stopping based on validation AUC

4. **Context-Aware Prediction** (`predict_with_context()` at main.py:160)
   - Groups test paragraphs by title (same document)
   - Applies context adjustment: 70% individual prediction + 30% document average
   - Accounts for consistency within the same document

### Data Flow
1. Load CSV data from `data/` directory (train.csv, test.csv)
2. Split training texts into individual paragraphs
3. Train models using cross-validation on paragraph level
4. Make predictions with document-level context adjustment
5. Output to `submission.csv` matching `sample_submission.csv` format

### Key Files
- `main.py` - Complete ML pipeline implementation
- `data/train.csv` - Training data with title, full_text, generated labels
- `data/test.csv` - Test data for prediction
- `data/sample_submission.csv` - Submission format template
- `submission.csv` - Generated predictions output
- `best_model_fold_N.pt` - Saved model checkpoints (one per fold)
- `DEVELOPMENT.md` - Development process and decisions log
- `simple_test.py` - Quick environment validation
- `test_model.py` - Sample model training for testing
- `install_packages.py` - Package installation helper

## Project Management and Git Workflow

### Automated Workflow Rules

**IMPORTANT**: When working on this project, ALWAYS use the automated Git workflow system for proper version control and logging.

#### Before Making Changes
```bash
# Start a new feature branch automatically
python project_manager.py start --feature "feature-name" "Description of the work"
```

#### After Completing Work
```bash
# Auto-commit changes and create structured commits
python project_manager.py complete --feature "feature-name"

# Deploy to main branch (merge and cleanup)
python project_manager.py deploy --feature "feature-name"
```

#### Quick Workflow (All-in-One)
```bash
# For simple changes - creates branch, commits, and merges automatically
python project_manager.py auto "Description of changes made"
```

#### Check Project Status
```bash
# View current Git status and project health
python project_manager.py status
```

### Modular Architecture (New Structure)

The project has been refactored into a modular structure:

```
src/
├── config.py           # Centralized configuration management
├── data_processor.py   # Data loading and preprocessing
├── model_trainer.py    # Model training and cross-validation
├── predictor.py        # Prediction and context adjustment
├── evaluator.py        # Metrics calculation and evaluation
└── utils.py           # Git automation and logging utilities

scripts/
├── train.py           # Training script
├── predict.py         # Prediction script
└── evaluate.py        # Evaluation script

models/                # Saved model checkpoints
logs/                  # Structured logging output
```

### Git Workflow Automation Features

1. **Feature Branch Creation**: Automatically creates `feature/name` branches
2. **Structured Commits**: Follows conventional commit format with Claude attribution
3. **Logging**: All actions logged to `logs/project_events.jsonl`
4. **Status Monitoring**: Real-time project health monitoring
5. **Auto-merge**: Safe merging to main branch with cleanup

### Command Guidelines

#### When Starting New Work:
- ALWAYS use `python project_manager.py start --feature "name" "description"`
- Choose descriptive feature names (e.g., "data-preprocessing", "model-optimization")

#### When Making Changes:
- Work normally in your feature branch
- The system tracks all file changes automatically

#### When Completing Work:
- Use `python project_manager.py complete --feature "name"` for complex features
- Use `python project_manager.py auto "description"` for simple changes

#### Logging and Monitoring:
- All Git operations are logged with timestamps
- System information is automatically captured
- Project events are stored in structured JSON format