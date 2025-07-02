# AI 텍스트 판별 챌린지 프로젝트

생성형 AI와 인간이 작성한 한국어 텍스트를 구별하는 BERT 기반 분류 모델입니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Python 가상환경 생성 (선택사항)
python3 -m venv rally_env
source rally_env/bin/activate  # Linux/Mac
# rally_env\Scripts\activate.bat  # Windows

# 필요한 패키지 설치
pip install -r requirements.txt

# 또는 설치 스크립트 사용
python3 install_packages.py
```

### 2. 데이터 검증

```bash
# 데이터 구조 확인
python3 test_data.py
```

### 3. 모델 학습 및 예측

```bash
# 전체 파이프라인 실행
python3 main.py
```

## 📁 프로젝트 구조

```
Rally/
├── main.py                 # 메인 실행 파일
├── requirements.txt        # 필요한 패키지 목록
├── install_packages.py     # 패키지 설치 스크립트
├── test_data.py           # 데이터 검증 스크립트
├── README.md              # 프로젝트 설명서
├── data/                  # 데이터 폴더
│   ├── train.csv          # 훈련 데이터 (1,226,364행)
│   ├── test.csv           # 테스트 데이터 (1,962행)
│   └── sample_submission.csv  # 제출 양식
└── submission.csv         # 생성된 예측 결과
```

## 🔧 모델 특징

### 핵심 구조
- **모델**: KLUE/BERT-base (한국어 특화)
- **방법**: 5-fold Cross Validation
- **배치 크기**: 8 (GPU 메모리 효율성)
- **학습률**: 2e-5 (Linear Warmup + Decay)

### 주요 기능
1. **문단 단위 분할**: 전체 텍스트를 문단별로 나누어 학습
2. **컨텍스트 인식**: 같은 문서 내 문단들의 일관성 고려
3. **메모리 최적화**: GPU 메모리 정리 및 배치 크기 조정
4. **대용량 데이터 처리**: 120만 행 이상의 데이터 효율적 처리

### 설정 변수
```python
MODEL_NAME = 'klue/bert-base'      # 한국어 BERT 모델
BATCH_SIZE = 8                     # 배치 크기
EPOCHS = 3                         # 학습 에포크
N_SPLITS = 5                       # K-Fold 분할 수
MAX_LENGTH = 512                   # 토큰 최대 길이
LEARNING_RATE = 2e-5               # 학습률
CONTEXT_WEIGHT = 0.3               # 컨텍스트 보정 가중치
```

## 📊 데이터 처리 과정

1. **전처리**: 전체 텍스트를 문단 단위로 분할
2. **토크나이징**: KLUE/BERT 토크나이저 사용
3. **학습**: 5-fold CV로 안정적인 성능 확보
4. **예측**: 문서별 컨텍스트를 고려한 조정된 예측값 생성

## 🎯 성능 지표

- **평가 메트릭**: ROC-AUC Score
- **검증 방법**: Out-of-Fold (OOF) 검증
- **컨텍스트 보정**: 70% 개별 예측 + 30% 문서 평균

## 💡 사용 팁

1. **GPU 사용 권장**: CUDA 지원 GPU가 있으면 훨씬 빠른 학습 가능
2. **메모리 부족 시**: `BATCH_SIZE`를 4로 줄여보세요
3. **빠른 테스트**: 코드 수정 후 `test_data.py`로 먼저 검증
4. **커스터마이징**: 상단의 설정 변수들을 수정하여 실험 가능

## 🏆 대회 규칙 준수

- ✅ 한국어 텍스트 처리
- ✅ 문단 단위 예측
- ✅ 동일 문서 내 컨텍스트 활용 허용
- ✅ 2025년 7월 이전 공개 모델 사용 (KLUE/BERT)
- ✅ 외부 API 미사용 (로컬 실행)

## 🔍 문제 해결

### 자주 발생하는 오류
1. **ModuleNotFoundError**: `pip install -r requirements.txt` 실행
2. **CUDA 메모리 부족**: `BATCH_SIZE` 줄이기
3. **인코딩 오류**: UTF-8-BOM 처리 완료
4. **데이터 경로 오류**: 상대 경로 사용으로 해결

실행 중 문제가 발생하면 `test_data.py`로 먼저 데이터를 확인해보세요!