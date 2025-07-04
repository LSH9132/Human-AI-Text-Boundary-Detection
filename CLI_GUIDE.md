# 🛠️ CLI 도구 사용 가이드

AI Text Detection 프로젝트의 모든 CLI 도구 사용법을 정리한 완전 가이드입니다.

## 📋 목차
- [환경 설정](#환경-설정)
- [메인 실행 스크립트](#메인-실행-스크립트)
- [개별 모듈 스크립트](#개별-모듈-스크립트)
- [Submission 관리 도구](#submission-관리-도구)
- [프로젝트 관리 도구](#프로젝트-관리-도구)
- [빠른 참조](#빠른-참조)

---

## 🔧 환경 설정

### 가상환경 활성화
```bash
# 가상환경 활성화 (필수)
source .venv/bin/activate

# 의존성 설치 (최초 1회)
pip install -r requirements.txt
```

### 환경 테스트
```bash
# 기본 의존성 테스트
python -c "import torch, transformers, sklearn, pandas; print('✅ All dependencies OK')"

# 모듈 import 테스트
python -c "from src import *; print('✅ All modules OK')"
```

---

## 🚀 메인 실행 스크립트

### main.py - 전체 ML 파이프라인

```bash
# 기본 실행 (CPU, 전체 파이프라인)
python main.py

# GPU 환경에서 실행
python main.py --env gpu

# CPU 최적화 실행
python main.py --env cpu

# 디버그 모드 (빠른 테스트, 1 epoch, 2-fold)
python main.py --env debug

# 로그 레벨 조정
python main.py --log-level DEBUG
python main.py --log-level WARNING

# 레거시 모드 강제 실행
python main.py --legacy
```

**실행 단계:**
1. 📊 데이터 로딩 및 검증
2. 🏋️ 모델 훈련 (5-fold 교차검증)
3. 🔮 예측 생성 (앙상블)
4. 💾 Submission 저장 (버전 관리)
5. 📈 평가 및 리포트 생성

---

## 📦 개별 모듈 스크립트

### scripts/train.py - 훈련 전용

```bash
# 기본 훈련
python scripts/train.py

# GPU 환경 훈련
python scripts/train.py --env gpu

# 디버그 모드 훈련
python scripts/train.py --env debug --log-level DEBUG
```

**출력물:**
- `models/best_model_fold_*.pt` (모델 체크포인트)
- `results/training_report.json` (훈련 결과)

### scripts/predict.py - 예측 전용

```bash
# 자동 모델 검색 예측
python scripts/predict.py

# 특정 모델 지정 예측
python scripts/predict.py --models models/best_model_fold_1.pt models/best_model_fold_2.pt

# 앙상블 방법 지정
python scripts/predict.py --ensemble --method mean
python scripts/predict.py --ensemble --method median

# 단일 모델 예측
python scripts/predict.py --models models/best_model_fold_1.pt

# 출력 파일명 지정
python scripts/predict.py --output my_submission.csv

# 상세 예측 정보 저장
python scripts/predict.py --save-detailed

# GPU 환경 예측
python scripts/predict.py --env gpu
```

**출력물:**
- `submission.csv` (또는 지정한 파일명)
- `results/prediction_report.json`
- `results/detailed_predictions.csv` (--save-detailed 사용시)

### scripts/evaluate.py - 평가 전용

```bash
# 예측 파일 분석 (Ground Truth 없이)
python scripts/evaluate.py --predictions submission.csv

# Ground Truth와 비교 평가
python scripts/evaluate.py --predictions submission.csv --labels data/validation.csv

# 임계값 조정
python scripts/evaluate.py --predictions submission.csv --threshold 0.6

# 상세 분석
python scripts/evaluate.py --predictions submission.csv --detailed

# 그래프 생성 (matplotlib 설치시)
python scripts/evaluate.py --predictions submission.csv --plots

# 결과 디렉토리 지정
python scripts/evaluate.py --predictions submission.csv --output-dir my_results
```

**출력물:**
- `results/evaluation_report.json`
- `results/detailed_evaluation.json` (--detailed 사용시)
- `results/plots/` (--plots 사용시)

---

## 📊 Submission 관리 도구

### submission_tool.py - Submission 버전 관리

#### 📋 목록 및 요약

```bash
# 모든 submission 목록
python submission_tool.py list

# 요약 정보
python submission_tool.py summary

# 최고 성능 submission 찾기
python submission_tool.py best
python submission_tool.py best --metric mean_prediction
```

**출력 예시:**
```
📋 Found 5 submissions:

Filename                                 Date                 Mean Pred  Description
==========================================================================================
submission_20250704_143022_c46ec7a.csv  2025-07-04 14:30:22  0.4523     debug environment - 2 fold ensemble
submission_20250704_150315_d57bf8b.csv  2025-07-04 15:03:15  0.4612     default environment - 5 fold ensemble
submission_20250704_162144_a89ce4d.csv  2025-07-04 16:21:44  0.4489     gpu environment - 5 fold ensemble
```

#### 🔍 비교 및 분석

```bash
# 두 submission 비교
python submission_tool.py compare submission_A.csv submission_B.csv

# 상세 리포트 생성
python submission_tool.py report
python submission_tool.py report --output my_report.json
```

**비교 출력 예시:**
```
🔍 Comparing submissions:
   File 1: submission_20250704_143022_c46ec7a.csv
   File 2: submission_20250704_150315_d57bf8b.csv
   Mean absolute difference: 0.0234
   Max absolute difference: 0.1567
   Correlation: 0.8912
   Samples with large diff (>0.1): 45
   Agreement rate (<0.05 diff): 89.2%
```

#### 🧹 관리 및 정리

```bash
# 오래된 submission 정리 (최신 10개 유지)
python submission_tool.py cleanup

# 최신 5개만 유지
python submission_tool.py cleanup --keep 5
```

---

## ⚙️ 프로젝트 관리 도구

### project_manager.py - Git 워크플로우 자동화

#### 🆕 새 기능 시작

```bash
# 새 기능 브랜치 생성
python project_manager.py start --feature "model-optimization" "Improve model performance with new architecture"

# 데이터 전처리 개선
python project_manager.py start --feature "data-preprocessing" "Add advanced text cleaning"
```

#### ✅ 작업 완료

```bash
# 기능 완료 및 커밋
python project_manager.py complete --feature "model-optimization"

# 특정 파일만 커밋
python project_manager.py complete --feature "model-optimization" --files src/model_trainer.py main.py
```

#### 🚀 배포

```bash
# 메인 브랜치에 배포
python project_manager.py deploy --feature "model-optimization"
```

#### 🎯 전체 자동화

```bash
# 간단한 변경사항 자동 처리 (브랜치 생성 → 커밋 → 배포)
python project_manager.py auto "Fix submission file format issue"

# 프로젝트 상태 확인
python project_manager.py status
```

**상태 출력 예시:**
```
📊 Project Status:
   Current branch: feature/model-optimization
   Latest commit: a1b2c3d
   Source files: 12
   Has changes: true
```

---

## ⚡ 빠른 참조

### 🎯 일반적인 워크플로우

#### 1️⃣ 전체 파이프라인 실행
```bash
source .venv/bin/activate
python main.py --env debug  # 빠른 테스트
python main.py              # 전체 실행
```

#### 2️⃣ 모델만 다시 훈련
```bash
python scripts/train.py --env gpu
```

#### 3️⃣ 기존 모델로 예측만
```bash
python scripts/predict.py --ensemble
```

#### 4️⃣ Submission 관리
```bash
python submission_tool.py list
python submission_tool.py compare sub1.csv sub2.csv
python submission_tool.py cleanup --keep 5
```

### 🔍 문제 해결

#### 의존성 문제
```bash
# 가상환경 재생성
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 모듈 Import 오류
```bash
# 모듈 경로 테스트
python -c "import sys; sys.path.append('src'); from src import config"
```

#### GPU 문제
```bash
# CPU 모드로 강제 실행
python main.py --env cpu
```

#### 메모리 부족
```bash
# 디버그 모드 (적은 데이터)
python main.py --env debug
```

### 📁 파일 구조 빠른 참조

```
🏠 프로젝트 루트/
├── 🐍 main.py              # 메인 실행 스크립트
├── 🛠️ submission_tool.py   # Submission 관리 CLI
├── ⚙️ project_manager.py   # Git 워크플로우 CLI
├── 📁 scripts/             # 개별 실행 스크립트
│   ├── train.py           # 훈련 전용
│   ├── predict.py         # 예측 전용
│   └── evaluate.py        # 평가 전용
├── 📁 src/                 # 핵심 모듈
├── 📁 submissions/         # Submission 버전 관리
├── 📁 models/              # 훈련된 모델
├── 📁 results/             # 결과 및 리포트
└── 📁 logs/                # 실행 로그
```

### ⌨️ 키보드 단축키 (추천)

터미널에서 자주 사용하는 명령어들:

```bash
# ~/.bashrc 또는 ~/.zshrc에 추가
alias aitrain="source .venv/bin/activate && python main.py"
alias aidebug="source .venv/bin/activate && python main.py --env debug"
alias aipredict="source .venv/bin/activate && python scripts/predict.py"
alias aisubmit="source .venv/bin/activate && python submission_tool.py"
alias aiproject="source .venv/bin/activate && python project_manager.py"

# 사용 예시
aitrain              # 전체 훈련
aidebug              # 디버그 모드
aisubmit list        # submission 목록
aiproject status     # 프로젝트 상태
```

---

## 💡 고급 팁

### 🔄 배치 처리

```bash
# 여러 환경에서 실험
for env in cpu gpu debug; do
    echo "Running with $env environment..."
    python main.py --env $env
    python submission_tool.py list | tail -1
done
```

### 📊 성능 모니터링

```bash
# 실시간 로그 모니터링
tail -f logs/main.log

# GPU 사용량 모니터링 (GPU 환경시)
watch -n 1 nvidia-smi
```

### 🎯 최적화된 실험

```bash
# 빠른 프로토타이핑
python main.py --env debug --log-level WARNING > quick_test.log 2>&1

# 상세한 분석
python main.py --env gpu --log-level DEBUG
python submission_tool.py summary
python submission_tool.py best
```

---

**📞 도움이 필요하신가요?**
- 각 스크립트에 `--help` 옵션 사용: `python main.py --help`
- CLAUDE.md 파일의 상세 가이드 참조
- 로그 파일 확인: `logs/` 디렉토리