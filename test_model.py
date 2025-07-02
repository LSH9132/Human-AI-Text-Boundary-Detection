#!/usr/bin/env python3
"""
빠른 모델 검증용 스크립트 - 소량의 데이터로 테스트
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# 설정 변수들 (테스트용으로 축소)
MODEL_NAME = 'klue/bert-base'
BATCH_SIZE = 4  # 더 작은 배치 크기
EPOCHS = 1  # 빠른 테스트를 위해 1 에포크만
N_SPLITS = 2  # 2-fold로 축소
MAX_LENGTH = 256  # 토큰 길이 축소
LEARNING_RATE = 2e-5
CONTEXT_WEIGHT = 0.3
SAMPLE_SIZE = 1000  # 1000개 샘플만 사용

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
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
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item

def load_sample_data():
    """샘플 데이터 로드"""
    print("샘플 데이터를 로드합니다...")
    train_df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
    test_df = pd.read_csv('data/test.csv', encoding='utf-8-sig')
    
    # 훈련 데이터 샘플링
    train_sample = train_df.sample(n=min(SAMPLE_SIZE, len(train_df)), random_state=42)
    
    # 테스트 데이터 샘플링
    test_sample = test_df.sample(n=min(100, len(test_df)), random_state=42)
    
    print(f"훈련 샘플: {len(train_sample)}개")
    print(f"테스트 샘플: {len(test_sample)}개")
    
    # 문단 분할 (간단화)
    train_paragraphs = []
    for idx, row in train_sample.iterrows():
        paragraphs = [p.strip() for p in row['full_text'].split('\n') if p.strip()]
        for p_idx, paragraph in enumerate(paragraphs[:3]):  # 최대 3개 문단만
            train_paragraphs.append({
                'title': row['title'],
                'paragraph_text': paragraph,
                'generated': row['generated']
            })
    
    train_para_df = pd.DataFrame(train_paragraphs)
    print(f"훈련 문단: {len(train_para_df)}개")
    
    return train_para_df, test_sample

def train_model(model, train_loader, val_loader, device, epochs=1, fold=0):
    """간단한 모델 훈련"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_auc = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].cpu().numpy()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()

                val_preds.extend(preds)
                val_labels.extend(labels)

        val_auc = roc_auc_score(val_labels, val_preds)
        print(f"Train Loss: {train_loss / len(train_loader):.4f}, Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc

    return best_auc

def main():
    print("=" * 50)
    print("AI 텍스트 판별 모델 - 빠른 검증")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")

    # 데이터 로드
    train_para_df, test_df = load_sample_data()

    # 토크나이저 초기화
    print("토크나이저를 로드합니다...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_para_df, train_para_df['generated'])):
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")

        # 데이터 분할
        train_texts = train_para_df.iloc[train_idx]['paragraph_text'].values
        train_labels = train_para_df.iloc[train_idx]['generated'].values
        val_texts = train_para_df.iloc[val_idx]['paragraph_text'].values
        val_labels = train_para_df.iloc[val_idx]['generated'].values

        # 데이터셋 생성
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        val_dataset = TextDataset(val_texts, val_labels, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 모델 초기화
        print("모델을 초기화합니다...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1
        ).to(device)

        # 학습
        best_auc = train_model(model, train_loader, val_loader, device, EPOCHS, fold)
        print(f"Fold {fold + 1} 최고 AUC: {best_auc:.4f}")
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        
        # 첫 번째 fold만 실행 (빠른 테스트)
        break

    print("\n✓ 모델 검증 완료!")
    print("전체 데이터로 학습하려면 main.py를 실행하세요.")

if __name__ == "__main__":
    main()