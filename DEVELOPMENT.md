# 개발 과정 기록 (Development Log)

## 프로젝트 개요
- **주제**: 생성형 AI와 인간 텍스트 판별 챌린지
- **모델**: KLUE/BERT 기반 한국어 텍스트 분류
- **목표**: 문단 단위 AI 생성 텍스트 판별 (ROC-AUC 최적화)

## 주요 개발 결정사항

### 1. 모델 아키텍처 선택
- **선택**: KLUE/BERT-base 모델
- **이유**: 한국어 텍스트에 특화된 사전훈련 모델
- **대안 고려**: KoBERT, RoBERTa-base
- **결정 근거**: 대회 규정 준수 (2025년 7월 이전 공개 모델)

### 2. 데이터 처리 전략
- **문제**: 전체 텍스트 라벨 → 문단별 예측 불일치
- **해결**: 전체 텍스트를 문단 단위로 분할하여 확장된 학습 데이터셋 생성
- **추가**: 같은 문서 내 문단들의 컨텍스트 일관성 활용

### 3. 컨텍스트 보정 알고리즘
```python
adjusted_prediction = 0.7 * individual_pred + 0.3 * document_average
```
- **근거**: 같은 문서 내 문단들은 유사한 생성 패턴을 가질 가능성
- **가중치**: 개별 예측 70% + 문서 평균 30%

## 핵심 개선사항

### 🔧 기술적 개선
1. **데이터 경로 수정**: `/data/` → `data/` (실행 오류 해결)
2. **Fold별 모델 저장**: `best_model_fold_N.pt` 형태로 분리
3. **메모리 최적화**: 
   - 배치 크기 16 → 8
   - GPU 메모리 정리 코드 추가
4. **학습률 스케줄러**: Linear warmup + decay 적용
5. **BOM 인코딩 처리**: UTF-8-sig로 한글 처리 문제 해결

### 📊 성능 최적화
1. **5-fold Cross Validation**: 안정적인 성능 평가
2. **Early Stopping**: 검증 AUC 기준 최적 모델 저장
3. **대용량 데이터 처리**: 120만+ 행 효율적 처리를 위한 진행률 표시

## 코드 구조 개선

### Before vs After
```python
# Before: 하드코딩된 설정
batch_size = 16
model_name = 'klue/bert-base'

# After: 설정 변수 분리
BATCH_SIZE = 8
MODEL_NAME = 'klue/bert-base'
```

### 모듈화된 구조
- `load_and_preprocess_data()`: 데이터 로딩 및 전처리
- `TextDataset`: PyTorch 데이터셋 클래스
- `train_model()`: 학습 로직
- `predict_with_context()`: 컨텍스트 인식 예측

## 검증 결과

### 환경 테스트 (.venv)
- ✅ Python 3.12.6 호환성 확인
- ✅ 모든 패키지 정상 설치 (pandas, torch, transformers)
- ✅ KLUE/BERT 모델 로딩 성공 (110M+ 파라미터)
- ✅ 한국어 토큰화 정상 작동
- ✅ 추론 엔진 검증 완료

### 데이터 검증
- **훈련 데이터**: 1,226,364행
- **테스트 데이터**: 1,962행
- **인코딩**: UTF-8-BOM 처리 완료
- **구조**: title, full_text, generated (train) / ID, title, paragraph_index, paragraph_text (test)

## 다음 개발 계획

### 1. CUDA 지원 추가
```python
# 계획된 개선사항
def get_optimal_device():
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"

def optimize_for_gpu():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return True
    return False
```

### 2. 모델 앙상블
- 여러 fold 모델들의 앙상블 예측
- 가중 평균 또는 스태킹 방법 고려

### 3. 하이퍼파라미터 튜닝
- 학습률, 배치 크기, 에포크 수 최적화
- 컨텍스트 가중치 조정

## 대회 규정 준수 체크리스트

- ✅ 한국어 텍스트 처리
- ✅ 문단 단위 예측
- ✅ 동일 문서 내 컨텍스트 활용 (허용됨)
- ✅ 2025년 7월 이전 공개 모델 사용
- ✅ 외부 API 미사용 (로컬 실행)
- ✅ Python 사용
- ✅ 외부 데이터 미사용

## 성능 기대치

### 예상 성능
- **Baseline AUC**: 0.8+ (BERT 기본 성능)
- **컨텍스트 보정 후**: 0.85+ 목표
- **5-fold 앙상블**: 0.87+ 기대

### 실행 시간 추정
- **CPU**: 4-6시간 (전체 데이터)
- **GPU**: 1-2시간 (CUDA 사용 시)

## 문제 해결 기록

### 1. 인코딩 문제
**문제**: UTF-8 BOM으로 인한 헤더 인식 오류
**해결**: `encoding='utf-8-sig'` 적용

### 2. 메모리 부족
**문제**: 대용량 데이터 처리 시 메모리 오버플로우
**해결**: 배치 크기 축소 + GPU 메모리 정리

### 3. 경로 오류
**문제**: 절대경로 사용으로 인한 실행 실패
**해결**: 상대경로로 변경

## 코드 리뷰 포인트

### 보안
- ✅ 악성 코드 없음
- ✅ 외부 API 호출 없음
- ✅ 로컬 파일만 접근

### 성능
- ✅ 메모리 효율적 데이터 로딩
- ✅ GPU 메모리 관리
- ✅ 배치 처리 최적화

### 유지보수성
- ✅ 설정 변수 분리
- ✅ 함수 모듈화
- ✅ 명확한 변수명

---

*이 문서는 Claude Code와의 개발 세션에서 자동 생성되었습니다.*
*최종 업데이트: 2025-07-02*