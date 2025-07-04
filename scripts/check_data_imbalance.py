#!/usr/bin/env python3
"""
데이터 불균형 분석 스크립트
Safe to run during training - read-only operations
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def analyze_class_distribution(train_df):
    """클래스 분포 분석"""
    print('📊 Train 데이터 클래스 분포 분석')
    print('=' * 50)
    
    total = len(train_df)
    print(f'총 샘플 수: {total:,}')
    
    # 클래스 분포
    class_counts = train_df['generated'].value_counts().sort_index()
    print(f'\n클래스별 샘플 수:')
    print(f'  Human (0): {class_counts[0]:,} ({class_counts[0]/total*100:.2f}%)')
    print(f'  AI (1): {class_counts[1]:,} ({class_counts[1]/total*100:.2f}%)')
    
    # 불균형 비율
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f'\n불균형 비율 (Human:AI): {imbalance_ratio:.2f}:1')
    
    # 심각도 평가
    print(f'\n📈 상세 분석:')
    print(f'  소수 클래스 비율: {min(class_counts)/total*100:.2f}%')
    print(f'  다수 클래스 비율: {max(class_counts)/total*100:.2f}%')
    
    if imbalance_ratio > 10:
        print(f'  ⚠️  심각한 불균형 (10:1 이상)')
    elif imbalance_ratio > 5:
        print(f'  ⚠️  중간 불균형 (5:1 이상)')
    else:
        print(f'  ✅ 비교적 균형')
    
    return class_counts, imbalance_ratio

def analyze_text_length_distribution(train_df):
    """텍스트 길이별 분포 분석"""
    print('\n📝 텍스트 길이별 클래스 분포')
    print('=' * 50)
    
    # 텍스트 길이 계산
    train_df['text_length'] = train_df['full_text'].str.len()
    
    # 길이별 통계
    print('평균 텍스트 길이:')
    human_avg = train_df[train_df['generated']==0]['text_length'].mean()
    ai_avg = train_df[train_df['generated']==1]['text_length'].mean()
    print(f'  Human: {human_avg:.0f} 글자')
    print(f'  AI: {ai_avg:.0f} 글자')
    print(f'  차이: {abs(human_avg - ai_avg):.0f} 글자')
    
    # 길이 구간별 분포
    bins = [0, 500, 1000, 2000, 5000, float('inf')]
    labels = ['0-500', '500-1K', '1K-2K', '2K-5K', '5K+']
    train_df['length_bin'] = pd.cut(train_df['text_length'], bins=bins, labels=labels, right=False)
    
    print('\n길이 구간별 클래스 분포:')
    for label in labels:
        subset = train_df[train_df['length_bin'] == label]
        if len(subset) > 0:
            total = len(subset)
            human_count = len(subset[subset['generated']==0])
            ai_count = len(subset[subset['generated']==1])
            human_pct = human_count/total*100
            ai_pct = ai_count/total*100
            print(f'  {label:>6}: Total {total:>6,} | Human {human_pct:>5.1f}% | AI {ai_pct:>5.1f}%')

def analyze_title_distribution(train_df):
    """제목별 분포 분석"""
    print('\n📚 제목별 샘플 수 분포')
    print('=' * 30)
    
    title_counts = train_df['title'].value_counts()
    print(f'총 고유 제목 수: {len(title_counts):,}')
    print(f'제목당 평균 샘플 수: {title_counts.mean():.1f}')
    print(f'최대 샘플 수: {title_counts.max()}')
    print(f'최소 샘플 수: {title_counts.min()}')
    
    # 중복 제목이 있는지 확인
    duplicated_titles = title_counts[title_counts > 1]
    if len(duplicated_titles) > 0:
        print(f'\n중복 제목 수: {len(duplicated_titles)}')
        print('상위 중복 제목들:')
        for title, count in duplicated_titles.head().items():
            print(f'  "{title[:50]}...": {count}개')
    else:
        print('\n모든 제목이 고유함 (중복 없음)')

def suggest_balancing_strategies(class_counts, imbalance_ratio):
    """불균형 처리 전략 제안"""
    print('\n⚖️ 불균형 처리 전략 분석')
    print('=' * 50)
    
    human_count = class_counts[0]
    ai_count = class_counts[1]
    
    print(f'현재 데이터:')
    print(f'  Human: {human_count:,} ({human_count/(human_count+ai_count)*100:.2f}%)')
    print(f'  AI: {ai_count:,} ({ai_count/(human_count+ai_count)*100:.2f}%)')
    print(f'  불균형 비율: {imbalance_ratio:.1f}:1')
    
    print(f'\n가능한 처리 방법들:')
    
    # 1. Undersampling
    balanced_size = ai_count * 2  # AI의 2배로 Human 줄이기
    print(f'1. Undersampling (2:1 비율):')
    print(f'   Human 샘플을 {balanced_size:,}개로 줄이기')
    print(f'   총 데이터: {balanced_size + ai_count:,}개')
    
    # 2. Oversampling  
    balanced_ai = human_count // 10  # 10:1 비율로 맞추기
    print(f'\n2. Oversampling (10:1 비율):')
    print(f'   AI 샘플을 {balanced_ai:,}개로 늘리기 (x{balanced_ai/ai_count:.1f})')
    print(f'   총 데이터: {human_count + balanced_ai:,}개')
    
    # 3. Class weights
    weight_ratio = human_count / ai_count
    print(f'\n3. Class Weight 조정:')
    print(f'   class_weight = {{0: 1.0, 1: {weight_ratio:.1f}}}')
    print(f'   또는 class_weight="balanced" 사용')
    
    print(f'\n💡 권장사항:')
    print(f'   • 현재 {imbalance_ratio:.1f}:1 불균형은 매우 심각한 수준')
    print(f'   • 문단 분할로 이미 데이터 확장 중')
    print(f'   • 추가로 class_weight 조정 권장')
    print(f'   • Stratified sampling 확인 필요')
    print(f'   • ROC-AUC 외에 PR-AUC도 모니터링')

def main():
    """메인 함수"""
    try:
        # 데이터 파일 경로
        data_path = Path('data/train.csv')
        
        if not data_path.exists():
            print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
            print("프로젝트 루트 디렉토리에서 실행해주세요.")
            return 1
        
        # 데이터 로드 (read-only)
        print("📁 데이터 로딩 중...")
        train_df = pd.read_csv(data_path)
        
        # 분석 실행
        class_counts, imbalance_ratio = analyze_class_distribution(train_df)
        analyze_text_length_distribution(train_df)
        analyze_title_distribution(train_df)
        suggest_balancing_strategies(class_counts, imbalance_ratio)
        
        print(f'\n✅ 분석 완료! 결과는 DATA_ANALYSIS.md에 문서화되어 있습니다.')
        return 0
        
    except Exception as e:
        print(f'❌ 분석 실패: {e}')
        return 1

if __name__ == "__main__":
    sys.exit(main())