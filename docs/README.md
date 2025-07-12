# 📚 프로젝트 문서 모음

이 디렉토리는 AI 텍스트 판별 프로젝트의 모든 문서를 정리한 곳입니다.

## 🚀 핵심 개선 문서 (최신)

### 1. 개선 가이드
- **[README_IMPROVED.md](./README_IMPROVED.md)** - 개선된 모델 사용법 및 실행 가이드
- **[IMPROVEMENT_LOG.md](./IMPROVEMENT_LOG.md)** - 전체 개선사항 종합 정리
- **[TECHNICAL_CHANGES.md](./TECHNICAL_CHANGES.md)** - 기술적 변경사항 상세 문서

### 2. 빠른 시작
```bash
# 추천 실행 방법
python train_improved.py --model koelectra --loss focal --env gpu

# 모델 테스트
python test_koelectra.py --compare
```

---

## 📋 기존 문서 아카이브

### 개발 과정
- **[DEVELOPMENT.md](./DEVELOPMENT.md)** - 개발 과정 기록 및 결정사항
- **[IMBALANCE_IMPROVEMENT_PLAN.md](./IMBALANCE_IMPROVEMENT_PLAN.md)** - 클래스 불균형 개선 계획

### 데이터 분석
- **[DATA_ANALYSIS.md](./DATA_ANALYSIS.md)** - 데이터 분석 결과
- **[COMPETITION.md](./COMPETITION.md)** - 대회 정보 및 규정

### 설정 및 환경
- **[CLI_GUIDE.md](./CLI_GUIDE.md)** - CLI 사용 가이드
- **[docker_test_report.md](./docker_test_report.md)** - Docker 테스트 결과

### README 파일들
- **[README_EN.md](./README_EN.md)** - 영문 README
- **[README_KR.md](./README_KR.md)** - 한국어 README

---

## 🎯 문서 우선순위

### 🔥 즉시 참고 (우선순위 높음)
1. **[README_IMPROVED.md](./README_IMPROVED.md)** - 개선된 실행 방법
2. **[IMPROVEMENT_LOG.md](./IMPROVEMENT_LOG.md)** - 무엇이 바뀌었는지
3. **[TECHNICAL_CHANGES.md](./TECHNICAL_CHANGES.md)** - 코드 변경 사항

### 📖 배경 지식 (참고용)
1. **[DEVELOPMENT.md](./DEVELOPMENT.md)** - 개발 히스토리
2. **[DATA_ANALYSIS.md](./DATA_ANALYSIS.md)** - 데이터 특성 이해
3. **[COMPETITION.md](./COMPETITION.md)** - 대회 규정

### 🛠️ 설정 및 환경 (필요 시)
1. **[CLI_GUIDE.md](./CLI_GUIDE.md)** - CLI 사용법
2. **[docker_test_report.md](./docker_test_report.md)** - Docker 환경

---

## 📊 성능 개선 요약

| 항목 | 이전 | 개선 후 |
|------|------|---------|
| **Public AUC** | 0.5 | 0.75+ (목표) |
| **모델** | KLUE/BERT | KoELECTRA |
| **손실 함수** | BCE | Focal Loss |
| **클래스 분포** | AI 8.2% | AI 30% (증강) |
| **교차 검증** | 문단별 | 문서별 |
| **앙상블** | 균등 가중치 | 최적화 가중치 |

---

## 🔄 문서 업데이트 히스토리

- **2025-01-09**: 프로젝트 구조 정리, docs 폴더 생성
- **2025-01-09**: 개선사항 문서 3개 추가
- **이전**: 개발 과정 문서들 (DEVELOPMENT.md 등)

---

## 💡 사용 팁

### 처음 사용하는 경우
1. **[README_IMPROVED.md](./README_IMPROVED.md)** 읽기
2. **[IMPROVEMENT_LOG.md](./IMPROVEMENT_LOG.md)** 에서 개선사항 확인
3. `python train_improved.py --model koelectra --loss focal --env gpu` 실행

### 문제 해결이 필요한 경우
1. **[README_IMPROVED.md](./README_IMPROVED.md)** 의 "문제 해결 가이드" 섹션
2. **[TECHNICAL_CHANGES.md](./TECHNICAL_CHANGES.md)** 에서 구체적 변경사항 확인
3. **[CLI_GUIDE.md](./CLI_GUIDE.md)** 에서 CLI 사용법 확인

### 개발 히스토리가 궁금한 경우
1. **[DEVELOPMENT.md](./DEVELOPMENT.md)** - 전체 개발 과정
2. **[DATA_ANALYSIS.md](./DATA_ANALYSIS.md)** - 데이터 분석 결과
3. **[COMPETITION.md](./COMPETITION.md)** - 대회 배경

---

## 📁 정리된 프로젝트 구조

```
Human-AI-Text-Boundary-Detection/
├── docs/                     # 📚 모든 문서 (여기!)
│   ├── README.md            # 문서 목차 (이 파일)
│   ├── README_IMPROVED.md   # 🚀 개선 가이드
│   ├── IMPROVEMENT_LOG.md   # 📋 개선사항 로그
│   ├── TECHNICAL_CHANGES.md # 🔧 기술적 변경사항
│   └── ... (기타 문서들)
├── src/                     # 핵심 소스 코드
├── data_augmentation/       # 데이터 증강 모듈
├── train_improved.py        # 개선된 훈련 스크립트
├── test_koelectra.py        # 모델 테스트 스크립트
├── data/                    # 데이터 파일
├── models/                  # 저장된 모델
├── logs/                    # 로그 파일
└── results/                 # 결과 파일
```

---

**🎯 시작하려면 [README_IMPROVED.md](./README_IMPROVED.md)를 확인하세요!**

*마지막 업데이트: 2025-01-09*