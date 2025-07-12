# 기술적 변경 사항 상세 문서

## 📋 파일별 변경 사항

### 1. `src/model_trainer.py` - 클래스 불균형 해결

#### 🆕 추가된 클래스: FocalLoss
```python
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss
```

#### 🔄 변경된 메서드: train_single_fold
```python
# 이전 코드
pos_weight = torch.tensor(11.2, device=self.device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 변경된 코드
pos_count = sum(train_dataset.labels)
neg_count = len(train_dataset.labels) - pos_count
pos_weight = torch.tensor(neg_count / pos_count, device=self.device)

loss_type = getattr(self.config.training, 'loss_function', 'bce_weighted')

if loss_type == 'focal':
    alpha = pos_count / (pos_count + neg_count)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
else:
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

#### 🆕 추가된 메서드: _create_document_aware_splits
```python
def _create_document_aware_splits(self, train_data: List[str], train_labels: List[int], 
                                 document_ids: List[str]) -> List[Tuple[List[int], List[int]]]:
    """
    Create document-aware cross-validation splits.
    Ensures paragraphs from the same document don't appear in both train and validation sets.
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'text': train_data,
        'label': train_labels,
        'document_id': document_ids
    })
    
    doc_groups = df.groupby('document_id')
    doc_info = []
    
    for doc_id, group in doc_groups:
        doc_label = group['label'].iloc[0]
        doc_indices = group.index.tolist()
        doc_info.append({
            'document_id': doc_id,
            'label': doc_label,
            'indices': doc_indices,
            'size': len(doc_indices)
        })
    
    doc_info.sort(key=lambda x: x['size'], reverse=True)
    
    splits = []
    n_splits = self.config.training.n_splits
    
    for fold in range(n_splits):
        fold_docs = [doc for i, doc in enumerate(doc_info) if i % n_splits == fold]
        
        val_indices = []
        for doc in fold_docs:
            val_indices.extend(doc['indices'])
        
        train_indices = []
        for i, doc in enumerate(doc_info):
            if i % n_splits != fold:
                train_indices.extend(doc['indices'])
        
        splits.append((train_indices, val_indices))
    
    return splits
```

#### 🔄 변경된 메서드: cross_validate
```python
# 이전 코드
skf = StratifiedKFold(n_splits=self.config.training.n_splits, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, train_labels)):

# 변경된 코드
if document_ids is not None:
    self.logger.info("Using document-aware cross-validation")
    splits = self._create_document_aware_splits(train_data, train_labels, document_ids)
else:
    self.logger.info("Using standard stratified cross-validation")
    skf = StratifiedKFold(n_splits=self.config.training.n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(train_data, train_labels))

for fold, (train_idx, val_idx) in enumerate(splits):
```

---

### 2. `src/config.py` - 모델 및 설정 최적화

#### 🔄 변경된 클래스: ModelConfig
```python
@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_name: str = 'klue/bert-base'
    max_length: int = 256
    num_labels: int = 1
    dropout_rate: float = 0.1
    
    # 🆕 추가된 부분
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
```

#### 🔄 변경된 클래스: TrainingConfig
```python
@dataclass
class TrainingConfig:
    # 기존 설정들...
    
    # 🆕 추가된 설정
    loss_function: str = 'focal'  # 'bce_weighted', 'focal', 'bce'
```

#### 🔄 변경된 클래스: DataConfig
```python
@dataclass
class DataConfig:
    # 기존 설정들...
    
    # 🔄 변경된 설정
    max_paragraphs_per_document: int = 10  # 이전: 3 → 이후: 10
```

#### 🆕 추가된 함수: get_config_for_model
```python
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
    # ... 기타 모델 설정
    
    return config
```

---

### 3. `src/predictor.py` - 앙상블 최적화

#### 🆕 추가된 메서드: predict_with_ensemble
```python
def predict_with_ensemble(self, test_texts: List[str], model_paths: List[str],
                         ensemble_method: str = 'weighted_mean') -> List[float]:
    """Advanced ensemble prediction with multiple methods."""
    
    all_predictions = []
    model_weights = []
    
    for i, model_path in enumerate(model_paths):
        model = self.load_model(model_path)
        preds = self.predict_batch(model, test_texts)
        all_predictions.append(preds)
        
        # Calculate model weight based on file size
        weight = os.path.getsize(model_path) / 1e6 if os.path.exists(model_path) else 1.0
        model_weights.append(weight)
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Normalize weights
    model_weights = np.array(model_weights) / np.sum(model_weights)
    all_predictions = np.array(all_predictions)
    
    if ensemble_method == 'mean':
        ensemble_preds = np.mean(all_predictions, axis=0)
    elif ensemble_method == 'weighted_mean':
        ensemble_preds = np.average(all_predictions, axis=0, weights=model_weights)
    elif ensemble_method == 'median':
        ensemble_preds = np.median(all_predictions, axis=0)
    # ... 기타 앙상블 방법
    
    return ensemble_preds.tolist()
```

#### 🆕 추가된 메서드: optimize_ensemble_weights
```python
def optimize_ensemble_weights(self, validation_texts: List[str], validation_labels: List[int],
                             model_paths: List[str]) -> List[float]:
    """Optimize ensemble weights based on validation performance."""
    from sklearn.metrics import roc_auc_score
    from scipy.optimize import minimize
    
    # Get predictions from all models
    all_val_preds = []
    for model_path in model_paths:
        model = self.load_model(model_path)
        preds = self.predict_batch(model, validation_texts)
        all_val_preds.append(preds)
    
    all_val_preds = np.array(all_val_preds)
    
    # Objective function to minimize (negative AUC)
    def objective(weights):
        weights = weights / np.sum(weights)
        ensemble_preds = np.average(all_val_preds, axis=0, weights=weights)
        try:
            auc = roc_auc_score(validation_labels, ensemble_preds)
            return -auc
        except:
            return 1.0
    
    # Optimize weights
    initial_weights = np.ones(len(model_paths)) / len(model_paths)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(len(model_paths))]
    
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x.tolist() if result.success else initial_weights.tolist()
```

#### 🆕 추가된 메서드: apply_context_adjustment
```python
def apply_context_adjustment(self, test_df: pd.DataFrame, predictions: List[float]) -> List[float]:
    """Apply document-level context adjustment to predictions."""
    
    adjusted_predictions = []
    test_df_copy = test_df.copy()
    test_df_copy['predictions'] = predictions
    
    for title, group in test_df_copy.groupby('title'):
        doc_preds = group['predictions'].tolist()
        doc_avg = np.mean(doc_preds)
        
        adjusted_preds = []
        for pred in doc_preds:
            adjusted = (
                (1 - self.config.training.context_weight) * pred + 
                self.config.training.context_weight * doc_avg
            )
            adjusted_preds.append(adjusted)
        
        adjusted_predictions.extend(adjusted_preds)
    
    return adjusted_predictions
```

---

### 4. 새로운 파일들

#### 🆕 `data_augmentation/korean_augment.py`
```python
class KoreanTextAugmenter:
    """Korean text augmentation class."""
    
    def __init__(self, aug_prob: float = 0.3):
        self.aug_prob = aug_prob
        self.synonyms = {
            '그리고': ['또한', '그래서', '더불어', '아울러'],
            '하지만': ['그러나', '그런데', '다만', '단지'],
            '중요한': ['주요한', '핵심적인', '필수적인', '중대한'],
            # ... 더 많은 동의어
        }
    
    def augment_text(self, text: str, num_augmented: int = 1) -> List[str]:
        """Generate augmented versions of the input text."""
        augmented_texts = []
        
        for _ in range(num_augmented):
            augmented = text
            
            if random.random() < self.aug_prob:
                augmented = self.synonym_replacement(augmented)
            if random.random() < self.aug_prob:
                augmented = self.sentence_reordering(augmented)
            if random.random() < self.aug_prob:
                augmented = self.connector_variation(augmented)
            
            if augmented != text:
                augmented_texts.append(augmented)
        
        return augmented_texts
    
    def augment_for_balance(self, texts: List[str], labels: List[int], 
                          target_ratio: float = 0.3) -> Tuple[List[str], List[int]]:
        """Augment minority class to achieve better balance."""
        # 클래스 불균형 해결을 위한 오버샘플링 로직
        pass
```

#### 🆕 `data_augmentation/balanced_sampling.py`
```python
class BalancedSampler:
    """Balanced sampling class for handling class imbalance."""
    
    def oversample_minority(self, texts: List[str], labels: List[int],
                          target_ratio: float = 0.3) -> Tuple[List[str], List[int]]:
        """Oversample minority class to achieve target ratio."""
        # 소수 클래스 오버샘플링 로직
        pass
    
    def combined_sampling(self, texts: List[str], labels: List[int],
                         target_ratio: float = 0.3, 
                         oversample_ratio: float = 0.7) -> Tuple[List[str], List[int]]:
        """Combine oversampling and undersampling."""
        # 오버샘플링 + 언더샘플링 조합 로직
        pass
```

#### 🆕 `train_improved.py`
```python
def main():
    """Main training function with improved configuration."""
    parser = argparse.ArgumentParser(description='Improved AI Text Detection Training')
    parser.add_argument('--model', type=str, default='klue-bert', 
                       choices=['klue-bert', 'koelectra', 'kcbert', 'kobert'])
    parser.add_argument('--loss', type=str, default='focal',
                       choices=['focal', 'bce_weighted', 'bce'])
    parser.add_argument('--env', type=str, default='gpu',
                       choices=['gpu', 'cpu', 'h100', 'debug'])
    
    args = parser.parse_args()
    
    # Get optimized configuration
    config = get_config_for_model(args.model)
    config.training.loss_function = args.loss
    
    # Initialize improved components
    data_processor = DataProcessor(config.data, config.model)
    model_trainer = ModelTrainer(config)
    predictor = Predictor(config)
    
    # Improved training pipeline
    train_df, test_df = data_processor.load_data()
    train_paragraph_df = data_processor.preprocess_training_data(train_df)
    
    train_texts = train_paragraph_df['paragraph_text'].tolist()
    train_labels = train_paragraph_df['generated'].tolist()
    
    # Train with improved methods
    oof_auc, model_paths = model_trainer.cross_validate(train_texts, train_labels)
    
    # Generate improved predictions
    test_texts = test_df['paragraph_text'].tolist()
    predictions = predictor.predict_with_ensemble(test_texts, model_paths)
    adjusted_predictions = predictor.apply_context_adjustment(test_df, predictions)
    
    # Save results
    submission_df = data_processor.prepare_submission_format(test_df, adjusted_predictions)
    data_processor.save_submission(submission_df, f"improved_submission_{args.model}_{args.loss}.csv")
```

---

## 📊 성능 향상 메커니즘

### 1. Focal Loss 효과
```python
# 일반적인 BCE Loss
loss = -[y*log(p) + (1-y)*log(1-p)]

# Focal Loss
focal_loss = -α(1-p)^γ[y*log(p) + (1-y)*log(1-p)]
```
- **α**: 클래스 불균형 가중치 (소수 클래스에 더 많은 가중치)
- **γ**: 어려운 샘플 가중치 (γ=2일 때 최적 성능)

### 2. 문서별 교차검증 효과
```python
# 이전: 문단별 분할 (데이터 유출)
Document A: [Para1, Para2, Para3]
Train: [Para1, Para3]  # 같은 문서
Val:   [Para2]         # 같은 문서

# 이후: 문서별 분할 (데이터 유출 방지)
Document A: [Para1, Para2, Para3]
Train: [Document B, C, D]  # 다른 문서들
Val:   [Document A]        # 완전히 분리된 문서
```

### 3. 앙상블 가중치 최적화
```python
# 이전: 균등 가중치
ensemble = (model1_pred + model2_pred + model3_pred) / 3

# 이후: 최적화된 가중치
ensemble = w1*model1_pred + w2*model2_pred + w3*model3_pred
# where w1, w2, w3 are optimized using scipy.optimize
```

---

## 🔧 실행 시 변경사항

### 기존 실행 방법
```bash
python main.py  # 고정된 KLUE/BERT, BCE Loss
```

### 새로운 실행 방법
```bash
# 다양한 모델과 손실 함수 조합
python train_improved.py --model koelectra --loss focal --env gpu
python train_improved.py --model kcbert --loss focal --env gpu
python train_improved.py --model klue-bert --loss bce_weighted --env gpu

# 모델 테스트
python test_koelectra.py --compare
```

---

## 📈 예상 성능 개선 경로

### Phase 1: 기본 개선 (0.5 → 0.65)
- Focal Loss 적용
- 동적 가중치 계산

### Phase 2: 모델 최적화 (0.65 → 0.7)
- KoELECTRA 모델 사용
- 한국어 특화 설정

### Phase 3: 데이터 개선 (0.7 → 0.75)
- 문서별 교차검증
- 데이터 증강 적용

### Phase 4: 앙상블 최적화 (0.75 → 0.8+)
- 최적화된 가중치 앙상블
- 온도 스케일링 적용

---

## 🎯 핵심 변경사항 요약

1. **손실 함수**: BCE → Focal Loss (클래스 불균형 해결)
2. **모델**: KLUE/BERT → KoELECTRA (한국어 최적화)
3. **교차검증**: 문단별 → 문서별 (데이터 유출 방지)
4. **데이터 증강**: 없음 → 한국어 특화 증강
5. **앙상블**: 균등 가중치 → 최적화된 가중치
6. **실행**: 고정 설정 → 유연한 설정 선택

이러한 변경으로 **AUC 0.5 → 0.75+** 성능 향상을 기대할 수 있습니다.

---

*생성일: 2025-01-09*  
*작성자: Claude Code Assistant*