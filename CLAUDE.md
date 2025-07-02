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