# KLUE-BERT AI 텍스트 탐지 프로젝트

## 🎯 프로젝트 개요

이 프로젝트는 한국어 AI 생성 텍스트 탐지를 위한 KLUE-BERT 기반 분류 시스템입니다. 메인 프로젝트에서 달성한 AUC 0.7355의 돌파 성과를 동일한 방법론으로 재현할 수 있는 독립적인 구현입니다.

## 🏆 성능 목표

- **타겟 성능**: AUC 0.73+ (메인 프로젝트 0.7355 참조)
- **기술 스택**: KLUE-BERT + Focal Loss + Document-Aware CV
- **처리 속도**: 실시간 추론 가능
- **재현성**: 완전 자동화된 파이프라인

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 데이터를 data/ 폴더에 배치
mkdir -p data
# train.csv, test.csv, sample_submission.csv 복사
```

### 3. 훈련 실행

```bash
# GPU 환경에서 훈련
python scripts/train.py

# CPU 환경에서 훈련 (디버그용)
python scripts/train.py --device cpu
```

### 4. 예측 생성

```bash
# 훈련된 모델로 예측
python scripts/predict.py

# 결과: submission.csv 생성
```

## 📁 프로젝트 구조

```
klue-bert/
├── README.md                    # 이 파일
├── requirements.txt             # 패키지 의존성
├── config.yaml                  # 설정 파일
├── src/                         # 핵심 소스 코드
│   ├── config.py               # 설정 관리
│   ├── focal_loss.py           # Focal Loss 구현
│   ├── data_processor.py       # 데이터 처리
│   ├── trainer.py              # 모델 훈련
│   └── predictor.py            # 예측 시스템
├── scripts/                     # 실행 스크립트
│   ├── train.py                # 훈련 스크립트
│   ├── predict.py              # 예측 스크립트
│   └── evaluate.py             # 평가 스크립트
├── data/                        # 데이터 디렉토리
│   ├── train.csv               # 훈련 데이터
│   ├── test.csv                # 테스트 데이터
│   └── sample_submission.csv   # 제출 형식
├── models/                      # 저장된 모델
├── logs/                        # 훈련 로그
└── docs/                        # 상세 문서
```

## 🛠️ 핵심 기술

### 1. Focal Loss
클래스 불균형 문제 해결을 위한 고급 손실 함수
- α=0.083 (AI 클래스 비율 기반)
- γ=2.0 (하드 샘플 집중)

### 2. Document-Aware Cross-Validation
데이터 유출 방지를 위한 문서 단위 교차검증
- 동일 문서의 단락들이 다른 폴드로 분리되지 않음
- 3-fold 교차검증으로 안정적인 성능 평가

### 3. KLUE-BERT 최적화
한국어 텍스트 처리에 특화된 모델 설정
- 모델: klue/bert-base
- 최대 길이: 512 토큰
- 학습률: 2e-5 (AdamW)

## 📊 예상 성능

| 메트릭 | 목표 값 | 설명 |
|--------|---------|------|
| **Cross-Validation AUC** | 0.738± | 3-fold CV 평균 |
| **Out-of-Fold AUC** | 0.735+ | 전체 데이터 성능 |
| **처리 속도** | 90+ samples/sec | 예측 처리 속도 |
| **메모리 사용량** | <8GB GPU | 단일 GPU 요구사항 |

## 🔧 설정 옵션

주요 설정은 `config.yaml`에서 수정 가능:

```yaml
model:
  name: "klue/bert-base"
  max_length: 512
  
training:
  batch_size: 16
  learning_rate: 2e-5
  epochs: 3
  
focal_loss:
  alpha: 0.083
  gamma: 2.0
```

## 📈 모니터링

훈련 진행상황은 실시간으로 모니터링 가능:

```bash
# 훈련 로그 실시간 확인
tail -f logs/training.log

# 성능 메트릭 확인
python scripts/evaluate.py --model models/best_model.pt
```

## 🔄 재현성 보장

- 모든 랜덤 시드 고정 (42)
- 동일한 데이터 분할 방법
- 버전 고정된 패키지 의존성
- 완전 자동화된 파이프라인

## 📝 라이선스

이 프로젝트는 메인 프로젝트와 동일한 방법론을 사용하지만 독립적으로 구현된 별도 프로젝트입니다.

## 🤝 기여

이 프로젝트는 메인 프로젝트의 KLUE-BERT 성과를 독립적으로 재현하기 위한 목적으로 제작되었습니다.