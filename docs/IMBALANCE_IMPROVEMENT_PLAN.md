# 📊 클래스 불균형 개선 계획

*Generated: 2025-07-04 19:35*  
*Status: 훈련 완료 후 실행 예정*

## 🎯 현재 상황 분석

### 📈 데이터 불균형 현황
- **클래스 분포**: 91.77% Human vs 8.23% AI (11.2:1)
- **현재 전략**: 문단 분할 (97K → 1.1M 샘플)
- **문제점**: 샘플 수는 늘었지만 **불균형 비율은 동일**
- **예상 성능**: CV 0.82-0.88, Public 0.78-0.85 (중간 수준)

### 🔍 현재 접근법의 한계
```python
# 현재 상황
원본: 97,172 문서 → 1,144,487 문단
불균형: 91.77% Human, 8.23% AI → 비율 변화 없음
효과: 데이터 확장 O, 불균형 해결 X
```

## 🚀 개선 전략 로드맵

### 1️⃣ **즉시 적용 가능한 방법들** (Priority 1)

#### A. Class Weight 조정
```python
# src/model_trainer.py 수정
class_weight = {
    0: 1.0,           # Human (다수 클래스)
    1: 11.2           # AI (소수 클래스 - 불균형 비율 반영)
}

# 또는 자동 계산
class_weight = "balanced"  # sklearn 스타일
```

#### B. Focal Loss 적용
```python
# src/model_trainer.py에 추가
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# 사용법
criterion = FocalLoss(alpha=0.25, gamma=2.0)  # 소수 클래스에 집중
```

#### C. 임계값 최적화
```python
# src/predictor.py에 추가
def find_optimal_threshold(y_true, y_pred_proba):
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold

# 사용법
optimal_threshold = find_optimal_threshold(y_val, y_pred_proba)
y_pred = (y_pred_proba > optimal_threshold).astype(int)
```

### 2️⃣ **중급 개선 방법들** (Priority 2)

#### A. SMOTE Oversampling
```python
# scripts/improve_imbalance.py 생성
from imblearn.over_sampling import SMOTE, ADASYN

def apply_smote_oversampling(X, y):
    """SMOTE를 사용한 오버샘플링"""
    smote = SMOTE(
        sampling_strategy='auto',  # 자동 균형 조정
        random_state=42,
        k_neighbors=5
    )
    
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"Before SMOTE: {Counter(y)}")
    print(f"After SMOTE: {Counter(y_resampled)}")
    
    return X_resampled, y_resampled

# ADASYN 대안
def apply_adasyn_oversampling(X, y):
    """ADASYN을 사용한 적응적 오버샘플링"""
    adasyn = ADASYN(
        sampling_strategy='auto',
        random_state=42,
        n_neighbors=5
    )
    
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled
```

#### B. Stratified Sampling 강화
```python
# src/model_trainer.py 수정
from sklearn.model_selection import StratifiedKFold

def enhanced_stratified_cv(X, y, n_splits=5):
    """개선된 계층화 교차검증"""
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # 각 fold에서 클래스 분포 확인
        train_dist = Counter(y[train_idx])
        val_dist = Counter(y[val_idx])
        
        print(f"Fold {fold+1} - Train: {train_dist}, Val: {val_dist}")
        
        yield train_idx, val_idx
```

### 3️⃣ **고급 데이터 증강** (Priority 3 - 규칙 확인 필요)

#### A. 안전한 텍스트 증강
```python
# scripts/text_augmentation.py 생성
import random
import re
from typing import List

class SafeKoreanAugmenter:
    """규칙 기반 한국어 텍스트 증강"""
    
    def __init__(self):
        # 한국어 동의어 사전 (수동 구축)
        self.synonyms = {
            "그러나": ["하지만", "그런데", "반면"],
            "따라서": ["그러므로", "때문에", "덕분에"],
            "또한": ["그리고", "더불어", "아울러"]
        }
        
        # 조사 변환 규칙
        self.particle_rules = {
            "는": "은", "이": "가", "를": "을"
        }
    
    def synonym_replacement(self, text: str, p: float = 0.1) -> str:
        """동의어 치환"""
        words = text.split()
        new_words = words.copy()
        
        for i, word in enumerate(words):
            if random.random() < p and word in self.synonyms:
                new_words[i] = random.choice(self.synonyms[word])
        
        return " ".join(new_words)
    
    def particle_change(self, text: str) -> str:
        """조사 변경"""
        for old, new in self.particle_rules.items():
            if old in text:
                text = text.replace(old, new)
        
        return text
    
    def sentence_shuffle(self, text: str) -> str:
        """문장 순서 변경"""
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 1:
            random.shuffle(sentences)
        
        return ". ".join(sentences) + "."
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """임의 단어 삭제"""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = [word for word in words if random.random() > p]
        
        # 최소 1개 단어는 유지
        if len(new_words) == 0:
            return random.choice(words)
        
        return " ".join(new_words)
    
    def augment_ai_texts(self, ai_texts: List[str], num_aug: int = 5) -> List[str]:
        """AI 텍스트만 증강"""
        augmented = []
        
        for text in ai_texts:
            # 원본 추가
            augmented.append(text)
            
            # 증강된 버전들 추가
            for _ in range(num_aug):
                aug_text = text
                
                # 여러 증강 기법 랜덤 적용
                if random.random() > 0.5:
                    aug_text = self.synonym_replacement(aug_text)
                if random.random() > 0.5:
                    aug_text = self.particle_change(aug_text)
                if random.random() > 0.3:
                    aug_text = self.sentence_shuffle(aug_text)
                if random.random() > 0.7:
                    aug_text = self.random_deletion(aug_text)
                
                augmented.append(aug_text)
        
        return augmented
```

#### B. BERT 기반 증강 (고급)
```python
# scripts/bert_augmentation.py
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

class KLUEBertAugmenter:
    """KLUE/BERT를 사용한 마스크 기반 증강"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.model = AutoModelForMaskedLM.from_pretrained("klue/bert-base")
        self.model.eval()
    
    def mask_and_predict(self, text: str, mask_prob: float = 0.15) -> str:
        """단어 마스킹 후 예측"""
        tokens = self.tokenizer.tokenize(text)
        
        # 랜덤하게 토큰 마스킹
        masked_tokens = []
        for token in tokens:
            if random.random() < mask_prob:
                masked_tokens.append('[MASK]')
            else:
                masked_tokens.append(token)
        
        # 마스크된 토큰 예측
        inputs = self.tokenizer(" ".join(masked_tokens), return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
        
        # 마스크 위치에 새로운 토큰 예측
        predicted_tokens = []
        for i, token in enumerate(masked_tokens):
            if token == '[MASK]':
                predicted_id = torch.argmax(predictions[0, i+1]).item()  # +1 for [CLS]
                predicted_token = self.tokenizer.decode([predicted_id])
                predicted_tokens.append(predicted_token)
            else:
                predicted_tokens.append(token)
        
        return self.tokenizer.convert_tokens_to_string(predicted_tokens)
```

## 🔧 실행 워크플로우

### Phase 1: 현재 훈련 완료 대기
```bash
# 현재 진행 상황 모니터링
tail -f gpu_training.log | grep -E "(AUC|Complete|Error)"

# 훈련 완료 시 결과 확인
python submission_tool.py summary
python scripts/check_data_imbalance.py
```

### Phase 2: 베이스라인 평가 (훈련 완료 즉시)
```bash
# 1. 현재 모델 성능 평가
python scripts/evaluate.py --predictions submission.csv --detailed

# 2. CV 점수와 실제 성능 비교 분석
python -c "
import pandas as pd
import json

# CV 결과 로드
with open('results/pipeline_report.json', 'r') as f:
    cv_results = json.load(f)

print(f'CV AUC: {cv_results[\"oof_auc\"]:.4f}')
print('Public 점수와 비교 분석 필요')
"

# 3. 불균형 영향 분석
python scripts/analyze_imbalance_impact.py  # 새로 생성 예정
```

### Phase 3: 즉시 개선 적용
```bash
# 1. Class Weight 버전 훈련
python main.py --env gpu --experiment class_weight

# 2. Focal Loss 버전 훈련  
python main.py --env gpu --experiment focal_loss

# 3. 두 방법 결합 버전
python main.py --env gpu --experiment combined
```

### Phase 4: 중급 개선 적용
```bash
# 1. SMOTE 오버샘플링
python scripts/train_with_smote.py --env gpu

# 2. 향상된 임계값 최적화
python scripts/optimize_threshold.py --input submission.csv

# 3. 앙상블 가중치 재조정
python scripts/reweight_ensemble.py --focus_minority_class
```

### Phase 5: 고급 증강 (규칙 허용 시)
```bash
# 1. 안전한 텍스트 증강
python scripts/augment_ai_texts.py --method safe --ratio 5

# 2. BERT 기반 증강
python scripts/augment_ai_texts.py --method bert --ratio 3

# 3. 최종 앙상블 훈련
python main.py --env gpu --experiment final_augmented
```

## 📋 실험 추적 템플릿

### 실험 결과 기록
```python
# experiments_log.json 업데이트
{
    "baseline": {
        "cv_auc": 0.xxxx,
        "public_auc": 0.xxxx,
        "private_auc": 0.xxxx,  # 대회 종료 후
        "method": "vanilla 5-fold CV",
        "class_ratio": "11.2:1"
    },
    "class_weight": {
        "cv_auc": 0.xxxx,
        "public_auc": 0.xxxx,
        "improvement": "+0.xxxx",
        "method": "balanced class weights",
        "config": {"weight_ratio": 11.2}
    },
    "focal_loss": {
        "cv_auc": 0.xxxx,
        "public_auc": 0.xxxx,
        "improvement": "+0.xxxx",
        "method": "focal loss",
        "config": {"alpha": 0.25, "gamma": 2.0}
    }
}
```

### 성능 비교 분석
```python
# scripts/compare_experiments.py
def analyze_improvement(baseline_auc, improved_auc):
    improvement = improved_auc - baseline_auc
    percentage = (improvement / baseline_auc) * 100
    
    print(f"절대 개선: +{improvement:.4f}")
    print(f"상대 개선: +{percentage:.2f}%")
    
    if improvement > 0.02:
        print("🎉 의미있는 개선!")
    elif improvement > 0.01:
        print("✅ 괜찮은 개선")
    else:
        print("⚠️ 미미한 개선")
```

## 🎯 성공 기준

### 단계별 목표
```python
# 현재 (예상)
baseline_auc = 0.82-0.88

# Phase 1 목표 (class weight + focal loss)
target_phase1 = baseline_auc + 0.02-0.04  # +2-4% 개선

# Phase 2 목표 (SMOTE + threshold optimization)  
target_phase2 = target_phase1 + 0.01-0.03  # 추가 +1-3% 개선

# Phase 3 목표 (advanced augmentation)
target_phase3 = target_phase2 + 0.01-0.02  # 추가 +1-2% 개선

# 최종 목표
final_target = 0.88-0.94  # CV AUC 기준
```

### 중단 조건
```python
# 다음 경우 추가 실험 중단:
1. 3회 연속 성능 개선 없음
2. CV와 Public 점수 차이 > 0.05 (과적합)
3. 훈련 시간 > 10시간 (효율성)
4. 메모리 사용량 > 90% (안정성)
```

## 🚨 주의사항

### 대회 규칙 준수
- [ ] 외부 API 사용 제한 확인
- [ ] 추가 데이터 생성 허용 여부 확인
- [ ] LLM 사용 가이드라인 확인
- [ ] 제출 횟수 제한 확인

### 기술적 주의사항
- [ ] 메모리 사용량 모니터링 (증강 시 급증 가능)
- [ ] 훈련 시간 관리 (증강 데이터로 인한 지연)
- [ ] 모델 저장 공간 확보 (여러 실험 버전)
- [ ] 로그 파일 정리 (용량 관리)

---

**📝 이 문서는 훈련 완료 후 체계적인 불균형 해결을 위한 로드맵입니다.**
**각 단계별로 신중하게 실험하며 성능 개선을 추적해야 합니다.**