# 🚀 AI 텍스트 판별 모델 개선 가이드

## 📋 개선 요약

**기존 성능**: Public AUC 0.5 (랜덤 수준)  
**개선 목표**: Public AUC 0.75+ (50% 향상)

### 🎯 주요 개선사항
1. **클래스 불균형 해결**: Focal Loss + 동적 가중치
2. **한국어 최적화**: KoELECTRA 모델 지원
3. **데이터 일관성**: 문단 단위 통일 (3→10개)
4. **교차검증 개선**: 문서별 분할 (데이터 유출 방지)
5. **데이터 증강**: 한국어 특화 증강 시스템
6. **앙상블 최적화**: 가중치 기반 조합

---

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 활성화
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate     # Windows

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 개선된 모델 훈련
```bash
# KoELECTRA + Focal Loss (권장)
python train_improved.py --model koelectra --loss focal --env gpu

# 다른 모델 조합 실험
python train_improved.py --model klue-bert --loss focal --env gpu
python train_improved.py --model kcbert --loss focal --env gpu
```

### 3. 모델 테스트
```bash
# KoELECTRA 기본 기능 테스트
python test_koelectra.py

# 여러 모델 비교
python test_koelectra.py --compare
```

---

## 🔧 실행 옵션

### train_improved.py 옵션
```bash
python train_improved.py [OPTIONS]

옵션:
  --model    모델 선택 (klue-bert, koelectra, kcbert, kobert)
  --loss     손실 함수 (focal, bce_weighted, bce)
  --env      환경 설정 (gpu, cpu, h100, debug)
  --dry-run  데이터 처리만 실행 (훈련 스킵)

예시:
  python train_improved.py --model koelectra --loss focal --env gpu
  python train_improved.py --model klue-bert --loss bce_weighted --env cpu
  python train_improved.py --dry-run  # 데이터 처리 테스트
```

### 모델별 최적화 설정
| 모델 | 배치 크기 | 학습률 | 최대 길이 | 손실 함수 |
|------|-----------|--------|-----------|-----------|
| KoELECTRA | 16 | 3e-5 | 512 | focal |
| KcBERT | 24 | 2e-5 | 300 | focal |
| KLUE-BERT | 32 | 2e-5 | 256 | focal |

---

## 📊 성능 비교

### 이전 vs 개선 후
| 항목 | 이전 | 개선 후 |
|------|------|---------|
| Public AUC | 0.5 | 0.75+ (목표) |
| 클래스 분포 | AI 8.2% | AI 30% (증강) |
| 손실 함수 | BCE (고정) | Focal (동적) |
| 모델 | KLUE/BERT | KoELECTRA |
| 교차검증 | 문단별 | 문서별 |
| 앙상블 | 균등 가중치 | 최적화 가중치 |

### 예상 성능 향상 단계
1. **0.5 → 0.65**: Focal Loss 적용
2. **0.65 → 0.7**: KoELECTRA 모델
3. **0.7 → 0.75**: 데이터 증강 + 문서별 CV
4. **0.75+**: 앙상블 최적화

---

## 🛠️ 핵심 개선 기술

### 1. Focal Loss
```python
# 클래스 불균형 해결
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        self.alpha = alpha  # 클래스 가중치
        self.gamma = gamma  # 어려운 샘플 가중치
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        return (self.alpha * focal_weight * bce_loss).mean()
```

### 2. 문서별 교차검증
```python
# 데이터 유출 방지
def document_aware_split(texts, labels, doc_ids):
    # 같은 문서의 문단들을 train/val에 섞지 않음
    unique_docs = set(doc_ids)
    for fold in range(n_folds):
        val_docs = [doc for i, doc in enumerate(unique_docs) if i % n_folds == fold]
        val_indices = [i for i, doc_id in enumerate(doc_ids) if doc_id in val_docs]
        train_indices = [i for i, doc_id in enumerate(doc_ids) if doc_id not in val_docs]
        yield train_indices, val_indices
```

### 3. 한국어 텍스트 증강
```python
# 클래스 균형 개선
augmenter = KoreanTextAugmenter()
balanced_texts, balanced_labels = augmenter.augment_for_balance(
    texts, labels, target_ratio=0.3
)

# 증강 기법: 동의어 교체, 문장 재배열, 연결어 변경
augmented = augmenter.augment_text("원본 텍스트", num_augmented=2)
```

---

## 📁 새로운 파일 구조

```
Human-AI-Text-Boundary-Detection/
├── src/                          # 핵심 모듈 (개선됨)
│   ├── model_trainer.py         # Focal Loss + 문서별 CV
│   ├── predictor.py             # 앙상블 최적화
│   └── config.py                # 모델별 설정
├── data_augmentation/           # 데이터 증강 (신규)
│   ├── korean_augment.py        # 한국어 특화 증강
│   ├── balanced_sampling.py     # 균형 샘플링
│   └── synthetic_generation.py  # 합성 데이터 생성
├── train_improved.py            # 개선된 훈련 스크립트
├── test_koelectra.py            # 모델 테스트 스크립트
├── IMPROVEMENT_LOG.md           # 개선 사항 로그
└── TECHNICAL_CHANGES.md         # 기술적 변경 사항
```

---

## 🔍 문제 해결 가이드

### 일반적인 문제들

#### 1. 환경 호환성 문제
```bash
# PyTorch 버전 확인
python -c "import torch; print(torch.__version__)"

# 필요 시 업그레이드
pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. 메모리 부족
```bash
# 배치 크기 줄이기
python train_improved.py --model koelectra --loss focal --env debug

# 또는 CPU 환경에서 실행
python train_improved.py --model koelectra --loss focal --env cpu
```

#### 3. 모델 로딩 실패
```bash
# 인터넷 연결 확인 후 재시도
python test_koelectra.py --compare

# 또는 캐시 삭제
rm -rf ~/.cache/huggingface/transformers/
```

### 성능 최적화 팁

#### 1. 하이퍼파라미터 튜닝
```python
# config.py에서 조정
config.training.learning_rate = 3e-5  # KoELECTRA에 최적
config.training.batch_size = 16       # 메모리에 맞게 조정
config.model.max_length = 512         # 긴 텍스트에 적합
```

#### 2. 데이터 증강 비율 조정
```python
# 클래스 균형 목표 조정
target_ratio = 0.3  # 30% AI 텍스트 (기본값)
target_ratio = 0.2  # 20% AI 텍스트 (보수적)
target_ratio = 0.4  # 40% AI 텍스트 (적극적)
```

#### 3. 앙상블 조합 최적화
```bash
# 다양한 모델 조합 실험
python train_improved.py --model koelectra --loss focal
python train_improved.py --model kcbert --loss focal
python train_improved.py --model klue-bert --loss bce_weighted

# 결과를 종합하여 최적 가중치 계산
```

---

## 📈 모니터링 및 평가

### 1. 훈련 모니터링
```python
# 로그 확인
tail -f improved_training.log

# 주요 지표
- Class distribution: 클래스 분포 개선 확인
- Fold AUC: 각 fold별 성능
- OOF AUC: 전체 out-of-fold 성능
- Model weights: 앙상블 가중치
```

### 2. 예측 결과 분석
```python
# 예측 분포 확인
submission = pd.read_csv('improved_submission_koelectra_focal.csv')
print(f"예측 평균: {submission['generated'].mean():.3f}")
print(f"0.5 이상 비율: {(submission['generated'] > 0.5).mean():.3f}")

# 실제 분포와 비교 (AI 텍스트 약 8.2%)
```

### 3. 성능 벤치마크
| 단계 | 예상 AUC | 구현 내용 |
|------|----------|-----------|
| Baseline | 0.5 | 기존 모델 |
| Phase 1 | 0.65 | Focal Loss |
| Phase 2 | 0.7 | KoELECTRA |
| Phase 3 | 0.75 | 데이터 증강 |
| Phase 4 | 0.8+ | 앙상블 최적화 |

---

## 🎯 추천 실행 순서

### 1. 기본 검증 (5분)
```bash
# 환경 및 데이터 확인
python train_improved.py --dry-run

# 모델 호환성 확인
python test_koelectra.py --compare
```

### 2. 빠른 실험 (30분)
```bash
# 디버그 모드로 빠른 검증
python train_improved.py --model koelectra --loss focal --env debug
```

### 3. 본격적인 훈련 (2-3시간)
```bash
# KoELECTRA + Focal Loss (권장)
python train_improved.py --model koelectra --loss focal --env gpu

# 추가 실험 (선택사항)
python train_improved.py --model kcbert --loss focal --env gpu
```

### 4. 결과 분석 및 제출
```bash
# 생성된 파일 확인
ls -la improved_submission_*.csv
ls -la improved_evaluation_report_*.json

# 최적 결과 선택하여 제출
```

---

## 🚀 기대 효과

### 정량적 개선
- **AUC**: 0.5 → 0.75+ (50% 향상)
- **클래스 균형**: 8.2% → 30% (3.6배 개선)
- **예측 정확도**: 랜덤 수준 → 실용적 수준

### 정성적 개선
- **모델 안정성**: 과적합 방지
- **일반화 성능**: 실제 테스트 데이터 대응
- **한국어 최적화**: 언어 특성 반영

---

## 📞 지원 및 문의

### 로그 파일 위치
- `improved_training.log`: 훈련 진행 로그
- `logs/project_events.jsonl`: 프로젝트 이벤트 로그
- `results/improved_evaluation_report_*.json`: 평가 결과

### 문제 해결 우선순위
1. **환경 설정**: PyTorch, CUDA 버전 확인
2. **메모리 관리**: 배치 크기 조정
3. **모델 로딩**: 인터넷 연결 및 캐시 확인
4. **데이터 처리**: 인코딩 및 경로 확인

---

## 🎉 성공 지표

### 단기 목표 (1주)
- [x] 개선된 시스템 구현 완료
- [ ] KoELECTRA 모델 성공적 훈련
- [ ] AUC 0.6+ 달성

### 중기 목표 (2주)
- [ ] 다양한 모델 조합 실험
- [ ] 데이터 증강 효과 검증
- [ ] AUC 0.7+ 달성

### 장기 목표 (1달)
- [ ] 앙상블 최적화 완료
- [ ] 하이퍼파라미터 튜닝 완료
- [ ] AUC 0.75+ 달성

---

**🚀 이제 `python train_improved.py --model koelectra --loss focal --env gpu` 명령으로 개선된 AI 텍스트 판별 모델을 시작하세요!**

*마지막 업데이트: 2025-01-09*