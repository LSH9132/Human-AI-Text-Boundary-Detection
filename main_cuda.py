#!/usr/bin/env python3
"""
AI 텍스트 판별 모델 - CUDA 최적화 버전
CUDA 사용 가능한 환경에서 GPU 가속을 활용한 고성능 학습
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import warnings
import gc

warnings.filterwarnings('ignore')


# ========================================
# CUDA 최적화 설정 변수들
# ========================================
MODEL_NAME = 'klue/bert-base'  # 한국어 BERT 모델
BATCH_SIZE = 16  # GPU에서 더 큰 배치 크기 사용
EPOCHS = 3  # 학습 에포크 수
N_SPLITS = 5  # K-Fold 교차검증 분할 수
MAX_LENGTH = 512  # 토큰 최대 길이
LEARNING_RATE = 2e-5  # 학습률
CONTEXT_WEIGHT = 0.3  # 컨텍스트 보정 가중치
MIXED_PRECISION = True  # Mixed Precision 사용 (GPU 메모리 절약)
GRADIENT_ACCUMULATION_STEPS = 2  # 그라디언트 누적 (효과적인 배치 크기 증가)


def setup_cuda():
    """CUDA 환경 설정 및 최적화"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA를 사용할 수 없습니다. CPU 버전(main.py)을 사용하세요.")
        return False, "cpu"
    
    # CUDA 최적화 설정
    torch.backends.cudnn.benchmark = True  # 고정 입력 크기에 대한 최적화
    torch.backends.cudnn.deterministic = False  # 성능 우선
    
    device = f"cuda:{torch.cuda.current_device()}"
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🚀 CUDA 환경 설정 완료!")
    print(f"   - GPU: {gpu_name}")
    print(f"   - 메모리: {gpu_memory:.1f}GB")
    print(f"   - 디바이스: {device}")
    print(f"   - Mixed Precision: {MIXED_PRECISION}")
    
    return True, device


def get_optimal_batch_size(device):
    """GPU 메모리에 따른 최적 배치 크기 결정"""
    if "cuda" not in device:
        return 8
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    if gpu_memory >= 24:  # RTX 4090, RTX 6000 Ada 등
        return 32
    elif gpu_memory >= 12:  # RTX 4070 Ti, RTX 3080 Ti 등
        return 24
    elif gpu_memory >= 8:   # RTX 4060 Ti, RTX 3070 등
        return 16
    else:  # RTX 3060, GTX 1660 등
        return 8


def clear_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# 1. 데이터 로드 및 전처리
def load_and_preprocess_data():
    """데이터 로드 및 문단 단위로 분할"""
    print("📊 데이터를 로드합니다...")
    # 학습 데이터 로드 (대용량 데이터 처리)
    train_df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
    test_df = pd.read_csv('data/test.csv', encoding='utf-8-sig')

    # 학습 데이터를 문단 단위로 분할
    print("✂️  텍스트를 문단 단위로 분할합니다...")
    train_paragraphs = []

    for idx, row in train_df.iterrows():
        if idx % 10000 == 0:
            print(f"   진행률: {idx:,}/{len(train_df):,} ({idx/len(train_df)*100:.1f}%)")
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

    print(f"✅ 데이터 로드 완료!")
    print(f"   - 원본 훈련 샘플: {len(train_df):,}개")
    print(f"   - 확장된 훈련 문단: {len(train_para_df):,}개")
    print(f"   - 테스트 문단: {len(test_df):,}개")

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


# 3. CUDA 최적화 모델 학습 함수
def train_model_cuda(model, train_loader, val_loader, device, epochs=3, fold=0):
    """CUDA 최적화된 모델 학습"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 학습률 스케줄러 설정
    total_steps = len(train_loader) * epochs // GRADIENT_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )

    # Mixed Precision 설정
    scaler = torch.cuda.amp.GradScaler() if MIXED_PRECISION else None

    best_auc = 0
    print(f"🎯 Fold {fold+1} 학습 시작 (Mixed Precision: {MIXED_PRECISION})")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            # Mixed Precision 사용
            if MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits.squeeze(), labels)
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits.squeeze(), labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # 진행률 출력
            if batch_idx % 100 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f}")

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].cpu().numpy()

                if MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        preds = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()

                val_preds.extend(preds)
                val_labels.extend(labels)

        val_auc = roc_auc_score(val_labels, val_preds)

        print(f"📈 Epoch {epoch + 1}/{epochs}")
        print(f"   Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"   Val AUC: {val_auc:.4f}")
        print(f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f'best_model_cuda_fold_{fold+1}.pt')
            print(f"   ✅ 새로운 최고 모델 저장! AUC: {best_auc:.4f}")

        # GPU 메모리 정리
        clear_gpu_memory()

    return best_auc


# 4. CUDA 최적화 컨텍스트 활용 예측 함수
def predict_with_context_cuda(model, test_df, tokenizer, device):
    """CUDA 최적화된 컨텍스트 활용 예측"""
    model.eval()
    predictions = []

    # 제목별로 그룹화
    grouped = test_df.groupby('title')
    total_groups = len(grouped)

    print(f"🔮 컨텍스트 예측 시작 ({total_groups}개 문서)")

    with torch.no_grad():
        for group_idx, (title, group) in enumerate(grouped):
            if group_idx % 100 == 0:
                print(f"   진행률: {group_idx}/{total_groups} ({group_idx/total_groups*100:.1f}%)")
            
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

                input_ids = encoding['input_ids'].to(device, non_blocking=True)
                attention_mask = encoding['attention_mask'].to(device, non_blocking=True)

                if MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        pred = torch.sigmoid(outputs.logits.squeeze()).cpu().item()
                else:
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
    print("=" * 60)
    print("🤖 AI 텍스트 판별 모델 - CUDA 가속 버전")
    print("=" * 60)
    
    # CUDA 환경 설정
    cuda_available, device = setup_cuda()
    if not cuda_available:
        return
    
    # GPU에 따른 배치 크기 최적화
    optimal_batch_size = get_optimal_batch_size(device)
    print(f"📦 최적화된 배치 크기: {optimal_batch_size}")
    
    # 설정 업데이트
    global BATCH_SIZE
    BATCH_SIZE = optimal_batch_size

    # 데이터 로드
    train_para_df, test_df = load_and_preprocess_data()

    # 토크나이저 및 모델 초기화
    print("🔧 토크나이저 로딩...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    oof_predictions = np.zeros(len(train_para_df))
    test_predictions = np.zeros(len(test_df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_para_df, train_para_df['generated'])):
        print(f"\n🔄 Fold {fold + 1}/{N_SPLITS} 시작")

        # 데이터 분할
        train_texts = train_para_df.iloc[train_idx]['paragraph_text'].values
        train_labels = train_para_df.iloc[train_idx]['generated'].values
        val_texts = train_para_df.iloc[val_idx]['paragraph_text'].values
        val_labels = train_para_df.iloc[val_idx]['generated'].values

        # 데이터셋 생성
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        val_dataset = TextDataset(val_texts, val_labels, tokenizer)

        # DataLoader에 CUDA 최적화 옵션 추가
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=2,  # GPU에서는 멀티프로세싱 사용
            pin_memory=True  # 메모리 고정으로 전송 속도 향상
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # 모델 초기화
        print("🧠 모델 초기화...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1
        ).to(device)

        # 학습
        best_auc = train_model_cuda(model, train_loader, val_loader, device, EPOCHS, fold)

        # 최고 모델 로드
        model.load_state_dict(torch.load(f'best_model_cuda_fold_{fold+1}.pt'))

        # OOF 예측
        model.eval()
        with torch.no_grad():
            val_preds = []
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)

                if MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        preds = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
                
                val_preds.extend(preds)

            oof_predictions[val_idx] = val_preds

        # 테스트 예측 (컨텍스트 활용)
        fold_test_preds = predict_with_context_cuda(model, test_df, tokenizer, device)
        test_predictions += np.array(fold_test_preds) / N_SPLITS
        
        print(f"✅ Fold {fold+1} 완료! 최고 AUC: {best_auc:.4f}")
        
        # GPU 메모리 정리
        clear_gpu_memory()

    # OOF 점수 계산
    oof_score = roc_auc_score(train_para_df['generated'], oof_predictions)
    print(f"\n🏆 최종 결과")
    print(f"   OOF AUC Score: {oof_score:.4f}")

    # 제출 파일 생성
    submission = pd.read_csv('data/sample_submission.csv', encoding='utf-8-sig')
    submission['generated'] = test_predictions
    submission.to_csv('submission_cuda.csv', index=False)
    print(f"📄 제출 파일 생성 완료: submission_cuda.csv")
    
    # 최종 GPU 메모리 상태
    if torch.cuda.is_available():
        print(f"💾 최대 GPU 메모리 사용량: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")


if __name__ == "__main__":
    main()