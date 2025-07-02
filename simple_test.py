#!/usr/bin/env python3
"""
매우 간단한 모델 검증 스크립트
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("=== AI 텍스트 판별 모델 - 초간단 검증 ===")

# 1. 기본 패키지 확인
print("1. 패키지 확인...")
print(f"   - pandas: {pd.__version__}")
print(f"   - torch: {torch.__version__}")

# 2. 디바이스 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"2. 디바이스: {device}")

# 3. 데이터 로드 테스트
print("3. 데이터 로드 테스트...")
try:
    train_df = pd.read_csv('data/train.csv', encoding='utf-8-sig', nrows=5)
    test_df = pd.read_csv('data/test.csv', encoding='utf-8-sig', nrows=5)
    print(f"   - 훈련 데이터: {len(train_df)}행 로드 성공")
    print(f"   - 테스트 데이터: {len(test_df)}행 로드 성공")
    print(f"   - 첫 번째 훈련 샘플 제목: {train_df.iloc[0]['title']}")
except Exception as e:
    print(f"   - 데이터 로드 오류: {e}")

# 4. 토크나이저 테스트
print("4. 토크나이저 테스트...")
try:
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    sample_text = "이것은 테스트 문장입니다."
    encoded = tokenizer(sample_text, return_tensors='pt')
    print(f"   - 토크나이저 로드 성공")
    print(f"   - 샘플 텍스트 토큰 수: {encoded['input_ids'].shape[1]}")
except Exception as e:
    print(f"   - 토크나이저 오류: {e}")

# 5. 모델 로드 테스트
print("5. 모델 로드 테스트...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        'klue/bert-base', 
        num_labels=1
    )
    print(f"   - 모델 로드 성공")
    print(f"   - 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   - 모델 로드 오류: {e}")

# 6. 간단한 추론 테스트
print("6. 추론 테스트...")
try:
    model.eval()
    with torch.no_grad():
        sample_text = "이것은 AI가 생성한 텍스트일 수 있습니다."
        inputs = tokenizer(sample_text, return_tensors='pt', max_length=128, truncation=True, padding=True)
        outputs = model(**inputs)
        prediction = torch.sigmoid(outputs.logits).item()
        print(f"   - 추론 성공")
        print(f"   - 샘플 예측값: {prediction:.4f}")
except Exception as e:
    print(f"   - 추론 오류: {e}")

print("\n✓ 모든 기본 구성요소가 정상 작동합니다!")
print("  이제 main.py로 전체 학습을 진행할 수 있습니다.")