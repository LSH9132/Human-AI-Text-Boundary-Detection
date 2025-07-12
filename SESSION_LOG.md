# Session Log - KoELECTRA vs KLUE-BERT 성능 분석 및 최종 앙상블

## 📅 세션 정보
- **날짜**: 2025-07-13
- **작업 범위**: KoELECTRA 모델 로딩 문제 해결, 앙상블 예측 완료, 성능 비교 분석
- **최종 결과**: KoELECTRA 앙상블 (AUC 0.706) 완료

## 🔍 주요 발견사항

### 1. KoELECTRA 모델 로딩 문제 해결

#### **문제**
- KoELECTRA 모델 파일(`best_model_fold_*.pt`)이 BERT 아키텍처로 로딩 시도
- `RuntimeError: size mismatch` 오류 발생

#### **근본 원인**
- 훈련 시 KoELECTRA 아키텍처로 저장된 모델 가중치
- 예측 시 KLUE-BERT 아키텍처로 로딩 시도
- 모델 파일명에 아키텍처 정보 부재

#### **해결책**
메타데이터 기반 자동 모델 감지 시스템 구현 (`src/predictor.py:31-80`):
```python
# 1. improved_training_metrics.json에서 모델 정보 읽기
# 2. 체크포인트 키 검사로 아키텍처 자동 감지
# 3. 올바른 아키텍처로 모델 로딩
```

### 2. KLUE-BERT vs KoELECTRA 성능 분석

#### **KLUE-BERT (성공 버전)**
- **모델**: `klue/bert-base`
- **손실 함수**: **BCEWithLogitsLoss** (일반 가중치)
- **Max Length**: 256
- **Batch Size**: 32
- **Max Paragraphs**: 3
- **성능**: AUC 0.7448 → 0.7355
- **예측 분포**: 균형적 (0.03-0.99), AI 분류율 27.5%

#### **KoELECTRA (현재 버전)**
- **모델**: `monologg/koelectra-base-v3-discriminator`
- **손실 함수**: **Focal Loss** (α=0.083, γ=2.0)
- **Max Length**: 512
- **Batch Size**: 16
- **Max Paragraphs**: 15
- **성능**: AUC 0.706
- **예측 분포**: 보수적 (0.11-0.65), AI 분류율 1.78%

#### **핵심 차이점**
1. **손실 함수**: KLUE-BERT는 일반 BCE, KoELECTRA는 Focal Loss
2. **예측 경향**: KLUE-BERT는 균형적, KoELECTRA는 과도하게 보수적
3. **데이터 처리**: KoELECTRA가 더 많은 컨텍스트 사용 (15 vs 3 paragraphs)

### 3. 최종 앙상블 결과

#### **KoELECTRA 3-Fold 앙상블**
- **모델 파일**: `models/best_model_fold_{1,2,3}.pt`
- **앙상블 방법**: weighted_mean
- **최종 제출**: `submissions/koelectra_final_submission.csv`

#### **예측 통계**
- **평균**: 0.1725
- **표준편차**: 0.040
- **범위**: 0.1131 - 0.6517
- **AI 분류율**: 1.78% (극도로 보수적)

## 📊 성능 비교 요약

| 지표 | KLUE-BERT | KoELECTRA |
|-----|----------|----------|
| **OOF AUC** | 0.7448 | 0.7056 |
| **최종 AUC** | 0.7355 | ~0.706 |
| **예측 평균** | 0.337 | 0.1725 |
| **AI 분류율** | 27.5% | 1.78% |
| **손실 함수** | BCE | Focal Loss |
| **Max Length** | 256 | 512 |

## 🔧 기술적 개선사항

### 1. 생성된 파일들
- `final_ensemble_prediction.py` - 인라인 코드의 스크립트 버전
- `submissions/koelectra_final_submission.csv` - 최종 제출 파일
- 수정된 `ensemble_predict.py` - 모델 탐지 패턴 개선

### 2. 메타데이터 기반 모델 감지
```python
# improved_training_metrics.json 활용
# 자동 아키텍처 감지 및 로딩
# 백워드 호환성 보장
```

### 3. 독립 KLUE-BERT 프로젝트
- `klue-bert/` - 완전히 독립된 구현
- 설정 기반 아키텍처 (`config.yaml`)
- 모듈화된 컴포넌트

## 🎯 결론 및 교훈

### 1. **Focal Loss의 양날의 검**
- 클래스 불균형 해결에 효과적이지만 과도한 보수성 초래 가능
- KLUE-BERT의 일반 BCE가 더 균형잡힌 결과 산출

### 2. **모델 아키텍처 호환성**
- 훈련과 예측 시 동일한 아키텍처 사용 필수
- 메타데이터 기반 자동 감지 시스템의 중요성

### 3. **하이퍼파라미터 영향**
- Max Length, Batch Size, Paragraph 수가 성능에 큰 영향
- 간단한 설정이 때로는 더 효과적

### 4. **앙상블 전략**
- 단일 모델 타입 앙상블도 충분히 효과적
- 모델 간 상호 보완보다는 안정성 향상에 기여

## 📈 향후 개선 방향

1. **손실 함수 최적화**: Focal Loss 파라미터 조정
2. **하이브리드 앙상블**: KLUE-BERT + KoELECTRA 조합
3. **컨텍스트 조정**: 문서 레벨 가중치 최적화
4. **데이터 증강**: 클래스 균형 개선 방법 탐구

## 📂 관련 파일
- `src/predictor.py` - 모델 로딩 및 앙상블 시스템
- `final_ensemble_prediction.py` - 재현 가능한 예측 스크립트
- `submissions/koelectra_final_submission.csv` - 최종 결과
- `models/improved_training_metrics.json` - KoELECTRA 메타데이터