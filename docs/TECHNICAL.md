# KLUE-BERT 기술 문서

## 🏗️ 아키텍처 개요

이 프로젝트는 메인 프로젝트에서 AUC 0.5 → 0.7355 돌파를 달성한 방법론을 독립적으로 구현합니다.

### 핵심 구성요소

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   데이터 처리   │───▶│    모델 훈련     │───▶│   예측 생성     │
│ KLUEDataProcessor│    │   KLUETrainer    │    │ KLUEPredictor   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ 단락 분할       │    │ Focal Loss       │    │ 앙상블 결합     │
│ 한국어 전처리   │    │ Document-Aware CV│    │ 배치 처리       │
│ 토큰화          │    │ 혼합 정밀도      │    │ 메모리 최적화   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔬 핵심 기술

### 1. Focal Loss

클래스 불균형 해결의 핵심:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.083, gamma=2.0):
        self.alpha = alpha    # AI 클래스 비율 (8.3%)
        self.gamma = gamma    # 어려운 샘플 집중도

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

**수학적 배경:**
- α = 0.083: AI 클래스 비율 기반 가중치
- γ = 2.0: 쉬운 샘플은 억제, 어려운 샘플은 강조
- PT = e^(-BCE): 올바른 예측 확률

### 2. Document-Aware Cross-Validation

데이터 유출 방지를 위한 교차검증:

```python
def create_document_aware_splits(self, processed_df):
    # 문서별 대표 레이블 계산
    doc_labels = processed_df.groupby('title')['generated'].agg(lambda x: x.mode().iloc[0])
    
    # GroupKFold로 문서 단위 분할
    group_kfold = GroupKFold(n_splits=self.config.cv.n_folds)
    
    for train_docs, val_docs in group_kfold.split(doc_labels.index, doc_labels.values, doc_labels.index):
        # 문서 → 단락 매핑
        train_titles = doc_labels.index[train_docs]
        val_titles = doc_labels.index[val_docs]
        
        train_indices = processed_df[processed_df['title'].isin(train_titles)].index
        val_indices = processed_df[processed_df['title'].isin(val_titles)].index
        
        yield train_indices, val_indices
```

**핵심 원칙:**
- 동일 문서의 단락들은 같은 폴드에 배치
- 문서 단위로 train/validation 분할
- 데이터 유출 완전 차단

### 3. 한국어 텍스트 전처리

KLUE-BERT에 최적화된 전처리:

```python
def clean_korean_text(self, text: str) -> str:
    # 한국어 친화적 특수문자 정제
    text = re.sub(r'[^\w\s가-힣.,!?;:\'"()[\]{}/-]', ' ', text)
    
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 한국어 구두점 정리
    text = re.sub(r'[.]{3,}', '...', text)
    
    return text.strip()

def split_into_paragraphs(self, text: str, title: str = "") -> List[str]:
    # 줄바꿈 기준 1차 분할
    lines = text.split('\n')
    
    # 짧은 문장들 병합 (한국어 특성 고려)
    paragraphs = []
    current_paragraph = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if len(current_paragraph) < 50:  # 한국어 최소 길이
            current_paragraph += " " + line if current_paragraph else line
        else:
            paragraphs.append(current_paragraph)
            current_paragraph = line
    
    # 토큰 길이 기반 재분할 (512 토큰 제한)
    filtered_paragraphs = []
    for p in paragraphs:
        if len(p) > self.config.model.max_length * 4:  # 한국어 평균 4글자/토큰
            # 문장 단위 재분할
            sentences = re.split(r'[.!?]\s+', p)
            # ... (청킹 로직)
    
    return filtered_paragraphs[:self.config.data.max_paragraphs_per_doc]
```

### 4. 앙상블 예측

다중 폴드 모델 결합:

```python
def predict_batch(self, dataset: KLUETextDataset) -> np.ndarray:
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_predictions = []
            
            # 각 폴드 모델에서 예측
            for model_idx, model in enumerate(self.models):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch.get('token_type_ids')
                )
                
                probs = torch.sigmoid(outputs.logits.squeeze(-1))
                weighted_probs = probs * self.ensemble_weights[model_idx]
                batch_predictions.append(weighted_probs.cpu().numpy())
            
            # 가중 평균 앙상블
            ensemble_pred = np.sum(batch_predictions, axis=0)
            all_predictions.extend(ensemble_pred)
    
    return np.array(all_predictions)
```

## 📊 성능 최적화

### 1. 메모리 최적화

```python
# 혼합 정밀도 훈련
scaler = GradScaler()
with autocast():
    outputs = model(input_ids, attention_mask)
    loss = focal_loss(outputs.logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 효율적 데이터로딩
DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    pin_memory=True,  # GPU 전송 최적화
    persistent_workers=True  # 워커 재사용
)
```

### 2. 훈련 안정화

```python
# 기울기 클리핑
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 학습률 스케줄링
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)

# 조기 종료
early_stopping = EarlyStopping(
    patience=3,
    min_delta=0.001,
    monitor='val_auc'
)
```

## 🔧 설정 시스템

### 계층적 설정 구조

```python
@dataclass
class Config:
    experiment_name: str = "klue_bert_detection"
    seed: int = 42
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    focal_loss: FocalLossConfig = field(default_factory=FocalLossConfig)
    # ...
```

### YAML 설정 예시

```yaml
model:
  name: "klue/bert-base"
  max_length: 512
  
training:
  batch_size: 16
  learning_rate: 2e-5
  epochs: 3
  
focal_loss:
  alpha: 0.083  # AI 클래스 비율
  gamma: 2.0    # 난이도 집중도
  
cross_validation:
  n_folds: 3
  strategy: "document_aware"
```

## 🧪 재현성 보장

### 시드 고정

```python
def setup_reproducibility(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 버전 고정

```requirements
torch==1.13.0
transformers==4.21.0
scikit-learn==1.1.0
# ... (모든 패키지 버전 고정)
```

## 📈 성능 벤치마크

| 메트릭 | 목표값 | 달성값 | 설명 |
|--------|--------|--------|------|
| CV AUC | 0.735+ | 0.738± | 3-fold 교차검증 평균 |
| 처리속도 | 50+ samples/sec | 90+ samples/sec | 예측 처리 속도 |
| GPU 메모리 | <8GB | ~6GB | 단일 GPU 요구사항 |
| 훈련시간 | <4시간 | ~3시간 | 전체 CV 훈련 |

## 🔍 디버깅 및 로깅

### 상세 로깅

```python
logger.info(f"📊 훈련 데이터: {len(train_df):,} 문서 → {len(processed_df):,} 단락")
logger.info(f"📈 클래스 분포: AI {ai_ratio:.1%}, Human {human_ratio:.1%}")
logger.info(f"🎯 Fold {fold} AUC: {auc:.4f}")
```

### 메트릭 추적

```python
# 에포크별 메트릭 저장
epoch_metrics = {
    'epoch': epoch,
    'train_loss': train_loss,
    'val_auc': val_auc,
    'val_accuracy': val_accuracy,
    'learning_rate': scheduler.get_last_lr()[0]
}
```

이 기술 문서는 메인 프로젝트의 성공 방법론을 완전히 독립적으로 구현한 결과입니다.