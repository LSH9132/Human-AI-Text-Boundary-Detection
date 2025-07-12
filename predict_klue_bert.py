#!/usr/bin/env python3
"""
Generate predictions using completed KLUE-BERT models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def predict_with_model(model_path, test_texts, tokenizer, device):
    """Generate predictions with a single model."""
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        'klue/bert-base',
        num_labels=1
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create dataset and dataloader
    dataset = SimpleTextDataset(test_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting with {os.path.basename(model_path)}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()
            predictions.extend(probs)
    
    return predictions

def main():
    print("🚀 KLUE-BERT Prediction Generation")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    
    # Load test data
    test_df = pd.read_csv('data/test.csv', encoding='utf-8-sig')
    test_texts = test_df['paragraph_text'].tolist()
    
    print(f"Test data loaded: {len(test_texts)} samples")
    print(f"Using device: {device}")
    
    # Model paths
    model_paths = [
        'models/klue_bert_gpu0_focal_v1/best_model_bert_base_fold_1.pt',
        'models/klue_bert_gpu0_focal_v1/best_model_bert_base_fold_2.pt', 
        'models/klue_bert_gpu0_focal_v1/best_model_bert_base_fold_3.pt'
    ]
    
    # Generate predictions for each model
    all_predictions = []
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            preds = predict_with_model(model_path, test_texts, tokenizer, device)
            all_predictions.append(preds)
            print(f"✅ Completed: {os.path.basename(model_path)}")
        else:
            print(f"❌ Model not found: {model_path}")
    
    if not all_predictions:
        print("❌ No models found!")
        return
    
    # Ensemble predictions (simple average)
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    print(f"Predictions generated: {len(ensemble_predictions)} samples")
    print(f"Prediction range: {ensemble_predictions.min():.4f} - {ensemble_predictions.max():.4f}")
    print(f"Mean prediction: {ensemble_predictions.mean():.4f}")
    
    # Save submission
    submission_df = pd.DataFrame({
        'ID': test_df['ID'],
        'generated': ensemble_predictions
    })
    
    submission_df.to_csv('klue_bert_submission.csv', index=False)
    print("💾 Submission saved to klue_bert_submission.csv")
    
    # Show sample predictions
    print("\n📊 Sample predictions:")
    for i in range(min(5, len(submission_df))):
        print(f"ID {submission_df.iloc[i]['ID']}: {submission_df.iloc[i]['generated']:.4f}")

if __name__ == "__main__":
    main()