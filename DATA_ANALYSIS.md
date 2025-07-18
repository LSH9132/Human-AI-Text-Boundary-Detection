# 📊 데이터 불균형 분석 리포트

*Generated: 2025-07-04 19:32*  
*Status: 훈련 중 (main.py --env gpu 실행 중)*

## 🔍 클래스 분포 분석

### 전체 데이터 개요
- **총 샘플 수**: 97,172개
- **고유 제목 수**: 97,172개 (제목당 1개 샘플)

### 클래스별 분포
| 클래스 | 샘플 수 | 비율 |
|--------|---------|------|
| Human (0) | 89,177 | 91.77% |
| AI (1) | 7,995 | 8.23% |

### 불균형 지표
- **불균형 비율**: 11.2:1 (Human:AI)
- **소수 클래스 비율**: 8.23%
- **심각도**: ⚠️ 심각한 불균형 (10:1 이상)

## 📝 텍스트 길이별 분석

### 평균 텍스트 길이
- **Human**: 2,325 글자
- **AI**: 2,299 글자
- **차이**: 미미함 (26글자)

### 길이 구간별 클래스 분포
| 길이 구간 | 총 샘플 | Human % | AI % |
|-----------|---------|---------|------|
| 0-500자 | 3 | 0.0% | 100.0% |
| 500-1K | 30,038 | 91.6% | 8.4% |
| 1K-2K | 37,499 | 91.8% | 8.2% |
| 2K-5K | 21,245 | 91.9% | 8.1% |
| 5K+ | 8,387 | 91.9% | 8.1% |

**관찰**: 모든 길이 구간에서 일관된 불균형 패턴 유지

## ⚖️ 불균형 처리 전략

### 현재 프로젝트에서 적용 중인 방법
1. **문단 단위 분할**: 97K → 1.1M 샘플 확장
2. **5-fold Cross Validation**: Stratified sampling
3. **Context-aware Prediction**: 문서 수준 일관성 적용

### 추가 개선 방안

#### 1. Undersampling (2:1 비율)
- Human 샘플을 15,990개로 줄이기
- 총 데이터: 23,985개
- **장점**: 빠른 훈련, 균형잡힌 학습
- **단점**: 정보 손실

#### 2. Oversampling (10:1 비율) 
- AI 샘플을 8,917개로 늘리기 (1.1배)
- 총 데이터: 98,094개
- **장점**: 정보 보존
- **단점**: 오버피팅 위험

#### 3. Class Weight 조정
```python
class_weight = {0: 1.0, 1: 11.2}  # 불균형 비율 반영
```

#### 4. Focal Loss 적용
```python
# 하드 샘플에 더 높은 가중치
alpha = 0.25, gamma = 2.0
```

#### 5. 앙상블 가중치 조정
```python
# 예측 시 소수 클래스 가중치 증가
```

## 🎯 권장사항

### 즉시 적용 가능 (다음 실험)
1. **Class Weight 추가**: `class_weight='balanced'`
2. **Stratified CV 확인**: 현재 적용 여부 검증
3. **Threshold 최적화**: ROC curve 기반 최적 임계값 찾기

### 중장기 개선
1. **Focal Loss 실험**: BCE 대신 Focal Loss 적용
2. **SMOTE 적용**: 오버샘플링 기법 테스트
3. **Ensemble 가중치**: 클래스별 모델 가중치 조정
4. **Data Augmentation**: 텍스트 증강 기법 적용

## 📈 예상 효과

### 현재 상황의 문제점
- **Precision 편향**: Human 클래스에 과도하게 치우침
- **Recall 저하**: AI 텍스트 탐지율 낮을 가능성
- **임계값 문제**: 0.5 기준이 부적절할 수 있음

### 개선 후 기대효과
- **Balanced F1-Score 향상**
- **AI 텍스트 탐지율 개선**
- **실제 운영 환경 성능 향상**

## ⚙️ 모니터링 지표

### 훈련 중 확인사항
- **Fold별 AUC 분산**: 과도한 분산 시 불균형 문제
- **Confusion Matrix**: 클래스별 성능 차이
- **Learning Curve**: 조기 수렴 여부

### 평가 지표 우선순위
1. **ROC-AUC**: 임계값 무관 성능
2. **PR-AUC**: 불균형 데이터에 더 민감
3. **F1-Score**: 정밀도-재현율 균형
4. **Balanced Accuracy**: 클래스별 가중 정확도

---

*이 분석은 훈련 중 안전하게 수행되었으며, 실행 중인 프로세스에 영향을 주지 않았습니다.*