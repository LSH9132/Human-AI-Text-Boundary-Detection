# KLUE-BERT 빠른 시작 가이드

## 🚀 5분 만에 시작하기

### 1. 환경 준비

```bash
# 프로젝트 디렉토리로 이동
cd klue-bert

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 데이터 디렉토리 생성
mkdir -p data

# 데이터 파일 복사 (메인 프로젝트에서)
cp ../data/train.csv data/
cp ../data/test.csv data/
cp ../data/sample_submission.csv data/
```

### 3. 기본 훈련

```bash
# 기본 설정으로 훈련 (약 3-4시간)
python scripts/train.py

# 빠른 테스트 (디버그 모드)
python scripts/train.py --debug
```

### 4. 예측 생성

```bash
# 훈련된 모델로 예측
python scripts/predict.py

# 결과 확인
ls -la submission.csv
```

### 5. 성능 평가

```bash
# 훈련 결과 분석
python scripts/evaluate.py

# 로그 확인
cat logs/training.log
```

## 📊 예상 결과

정상적으로 실행되면 다음과 같은 성과를 기대할 수 있습니다:

- **CV AUC**: 0.735+ (목표)
- **훈련 시간**: 약 3-4시간 (GPU 기준)
- **메모리 사용량**: 8GB GPU 메모리

## 🛠️ 설정 커스터마이징

`config.yaml` 파일을 수정하여 설정을 변경할 수 있습니다:

```yaml
# 배치 크기 조정 (메모리에 따라)
training:
  batch_size: 16  # GPU 메모리가 부족하면 8로 감소

# 폴드 수 조정
cross_validation:
  n_folds: 3  # 시간을 절약하려면 2로 감소

# Focal Loss 파라미터 튜닝
focal_loss:
  alpha: 0.083  # 클래스 비율에 따라 조정
  gamma: 2.0    # 어려운 샘플 집중도
```

## 🐛 문제 해결

### GPU 메모리 부족
```bash
# 배치 크기 감소
python scripts/train.py --config config.yaml
# config.yaml에서 batch_size를 8 또는 4로 설정
```

### 데이터 파일 없음
```bash
# 데이터 경로 확인
ls -la data/
# 필요한 파일: train.csv, test.csv, sample_submission.csv
```

### 모델 다운로드 실패
```bash
# 인터넷 연결 확인 및 Hugging Face 모델 다운로드
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('klue/bert-base')"
```

## 📈 성능 모니터링

### 실시간 로그 확인
```bash
# 훈련 진행상황 모니터링
tail -f logs/training.log
```

### GPU 사용량 확인
```bash
# NVIDIA GPU 사용량
nvidia-smi

# 실시간 모니터링
watch -n 1 nvidia-smi
```

## 🎯 다음 단계

1. **하이퍼파라미터 튜닝**: `config.yaml`에서 학습률, 배치 크기 등 조정
2. **데이터 증강**: 추가 데이터 전처리 기법 적용
3. **앙상블 확장**: 다른 한국어 모델과 결합
4. **성능 분석**: 상세한 오류 분석 및 개선점 도출

성공적인 훈련을 위해 충분한 GPU 메모리와 시간을 확보하세요!