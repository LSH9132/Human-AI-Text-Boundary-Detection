import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


# ========================================
# 설정 변수들
# ========================================
MODEL_NAME = 'klue/bert-base'  # 한국어 BERT 모델
BATCH_SIZE = 8  # 배치 크기 (GPU 메모리 효율성 고려)
EPOCHS = 3  # 학습 에포크 수
N_SPLITS = 5  # K-Fold 교차검증 분할 수
MAX_LENGTH = 512  # 토큰 최대 길이
LEARNING_RATE = 2e-5  # 학습률
CONTEXT_WEIGHT = 0.3  # 컨텍스트 보정 가중치 (0.7 개별 + 0.3 평균)


# 1. 데이터 로드 및 전처리
def load_and_preprocess_data():
    """데이터 로드 및 문단 단위로 분할"""
    print("데이터를 로드합니다...")
    # 학습 데이터 로드 (대용량 데이터 처리)
    train_df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
    test_df = pd.read_csv('data/test.csv', encoding='utf-8-sig')

    # 학습 데이터를 문단 단위로 분할
    print("텍스트를 문단 단위로 분할합니다...")
    train_paragraphs = []

    for idx, row in train_df.iterrows():
        if idx % 10000 == 0:
            print(f"진행률: {idx}/{len(train_df)}")
        title = row['title']
        full_text = row['full_text']
        label = row['generated']

        # 문단 분할 (빈 줄 기준)
        paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]

        for p_idx, paragraph in enumerate(paragraphs):
            train_paragraphs.append({
                'title': title,
                'paragraph_index': p_idx,
                'paragraph_text': paragraph,
                'generated': label,
                'original_idx': idx
            })

    train_para_df = pd.DataFrame(train_paragraphs)

    print(f"Original train samples: {len(train_df)}")
    print(f"Train paragraphs: {len(train_para_df)}")
    print(f"Test paragraphs: {len(test_df)}")

    return train_para_df, test_df


# 2. 데이터셋 클래스
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


# 3. 모델 학습 함수
def train_model(model, train_loader, val_loader, device, epochs=3, fold=0):
    """모델 학습"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 학습률 스케줄러 설정
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )

    best_auc = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        for batch in train_loader:
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

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pt')

    return best_auc


# 4. 컨텍스트 활용 예측 함수
def predict_with_context(model, test_df, tokenizer, device):
    """같은 제목의 문단들을 함께 고려하여 예측"""
    model.eval()
    predictions = []

    # 제목별로 그룹화
    grouped = test_df.groupby('title')

    with torch.no_grad():
        for title, group in grouped:
            # 같은 제목의 모든 문단 예측
            paragraphs = group['paragraph_text'].values

            group_preds = []
            for para in paragraphs:
                encoding = tokenizer(
                    para,
                    truncation=True,
                    padding='max_length',
                    max_length=MAX_LENGTH,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.sigmoid(outputs.logits.squeeze()).cpu().item()
                group_preds.append(pred)

            # 컨텍스트 보정 (같은 글의 문단들은 비슷한 경향)
            avg_pred = np.mean(group_preds)
            adjusted_preds = []

            for pred in group_preds:
                # 평균값을 고려한 보정
                adjusted = (1 - CONTEXT_WEIGHT) * pred + CONTEXT_WEIGHT * avg_pred
                adjusted_preds.append(adjusted)

            predictions.extend(adjusted_preds)

    return predictions


# 5. 메인 실행 함수
def main():
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 로드
    train_para_df, test_df = load_and_preprocess_data()

    # 토크나이저 및 모델 초기화
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    oof_predictions = np.zeros(len(train_para_df))
    test_predictions = np.zeros(len(test_df))

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
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1
        ).to(device)

        # 학습
        best_auc = train_model(model, train_loader, val_loader, device, EPOCHS, fold)

        # GPU 메모리 정리
        torch.cuda.empty_cache()
        
        # 최고 모델 로드
        model.load_state_dict(torch.load(f'best_model_fold_{fold+1}.pt'))

        # OOF 예측
        model.eval()
        with torch.no_grad():
            val_preds = []
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
                val_preds.extend(preds)

            oof_predictions[val_idx] = val_preds

        # 테스트 예측 (컨텍스트 활용)
        fold_test_preds = predict_with_context(model, test_df, tokenizer, device)
        test_predictions += np.array(fold_test_preds) / N_SPLITS
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()

    # OOF 점수 계산
    oof_score = roc_auc_score(train_para_df['generated'], oof_predictions)
    print(f"\n최종 OOF AUC Score: {oof_score:.4f}")

    # 제출 파일 생성
    submission = pd.read_csv('data/sample_submission.csv', encoding='utf-8-sig')
    submission['generated'] = test_predictions
    submission.to_csv('submission.csv', index=False)
    print("\n제출 파일 생성 완료!")


if __name__ == "__main__":
    main()