#!/usr/bin/env python3
"""
클래스 불균형 영향 분석 스크립트
훈련 완료 후 베이스라인 모델의 불균형 영향을 상세히 분석합니다.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def analyze_prediction_distribution(submission_file: str):
    """예측값 분포 분석"""
    print("📊 예측값 분포 분석")
    print("=" * 50)
    
    try:
        df = pd.read_csv(submission_file)
        predictions = df['generated'].values
        
        print(f"총 예측 샘플: {len(predictions):,}")
        print(f"예측 범위: {predictions.min():.4f} ~ {predictions.max():.4f}")
        print(f"예측 평균: {predictions.mean():.4f}")
        print(f"예측 표준편차: {predictions.std():.4f}")
        
        # 임계값별 분류 결과
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        print(f"\n📈 임계값별 AI 분류 비율:")
        
        for threshold in thresholds:
            ai_ratio = (predictions > threshold).mean()
            print(f"  임계값 {threshold}: {ai_ratio:.2%} AI 분류")
        
        # 예측값 구간별 분포
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(predictions, bins=bins)
        
        print(f"\n📊 예측값 구간별 분포:")
        for i, (start, end, count) in enumerate(zip(bins[:-1], bins[1:], hist)):
            percentage = count / len(predictions) * 100
            print(f"  {start:.1f}-{end:.1f}: {count:,}개 ({percentage:.1f}%)")
            
        return {
            'total_samples': len(predictions),
            'mean_prediction': float(predictions.mean()),
            'std_prediction': float(predictions.std()),
            'min_prediction': float(predictions.min()),
            'max_prediction': float(predictions.max()),
            'threshold_analysis': {str(t): float((predictions > t).mean()) for t in thresholds},
            'distribution_bins': {f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(count) for i, count in enumerate(hist)}
        }
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return None

def analyze_class_imbalance_impact():
    """클래스 불균형이 모델에 미친 영향 분석"""
    print("\n⚖️ 클래스 불균형 영향 분석")
    print("=" * 50)
    
    # 원본 훈련 데이터 불균형 확인
    try:
        train_df = pd.read_csv("data/train.csv")
        class_counts = train_df['generated'].value_counts()
        total = len(train_df)
        
        human_ratio = class_counts[0] / total
        ai_ratio = class_counts[1] / total
        imbalance_ratio = class_counts[0] / class_counts[1]
        
        print(f"📋 훈련 데이터 불균형:")
        print(f"  Human: {class_counts[0]:,}개 ({human_ratio:.2%})")
        print(f"  AI: {class_counts[1]:,}개 ({ai_ratio:.2%})")
        print(f"  불균형 비율: {imbalance_ratio:.1f}:1")
        
        # 불균형의 심각도 평가
        if imbalance_ratio > 10:
            severity = "심각"
            expected_impact = "높음"
        elif imbalance_ratio > 5:
            severity = "중간"
            expected_impact = "보통"
        else:
            severity = "경미"
            expected_impact = "낮음"
        
        print(f"\n📊 불균형 심각도: {severity}")
        print(f"📈 예상 성능 영향: {expected_impact}")
        
        return {
            'human_count': int(class_counts[0]),
            'ai_count': int(class_counts[1]),
            'human_ratio': float(human_ratio),
            'ai_ratio': float(ai_ratio),
            'imbalance_ratio': float(imbalance_ratio),
            'severity': severity,
            'expected_impact': expected_impact
        }
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return None

def analyze_model_bias(submission_file: str):
    """모델 편향성 분석"""
    print("\n🎯 모델 편향성 분석")
    print("=" * 50)
    
    try:
        df = pd.read_csv(submission_file)
        predictions = df['generated'].values
        
        # 1. 전반적 편향 분석
        mean_pred = predictions.mean()
        if mean_pred < 0.3:
            bias_direction = "Human 편향 (강함)"
            bias_level = "심각"
        elif mean_pred < 0.4:
            bias_direction = "Human 편향 (보통)"
            bias_level = "보통"
        elif mean_pred > 0.7:
            bias_direction = "AI 편향 (강함)"
            bias_level = "심각"
        elif mean_pred > 0.6:
            bias_direction = "AI 편향 (보통)"
            bias_level = "보통"
        else:
            bias_direction = "균형적"
            bias_level = "양호"
        
        print(f"📊 전반적 편향: {bias_direction}")
        print(f"📈 편향 수준: {bias_level}")
        
        # 2. 극단값 분석
        extreme_human = (predictions < 0.1).sum()
        extreme_ai = (predictions > 0.9).sum()
        moderate_range = ((predictions >= 0.3) & (predictions <= 0.7)).sum()
        
        print(f"\n📊 예측 극단성 분석:")
        print(f"  매우 확실한 Human (<0.1): {extreme_human:,}개 ({extreme_human/len(predictions):.1%})")
        print(f"  매우 확실한 AI (>0.9): {extreme_ai:,}개 ({extreme_ai/len(predictions):.1%})")
        print(f"  애매한 구간 (0.3-0.7): {moderate_range:,}개 ({moderate_range/len(predictions):.1%})")
        
        # 3. 신뢰도 분석
        confidence_scores = np.abs(predictions - 0.5) * 2  # 0.5에서 거리 * 2
        avg_confidence = confidence_scores.mean()
        
        print(f"\n🎯 모델 신뢰도:")
        print(f"  평균 신뢰도: {avg_confidence:.3f}")
        
        if avg_confidence > 0.8:
            confidence_level = "매우 높음 (과신 위험)"
        elif avg_confidence > 0.6:
            confidence_level = "높음"
        elif avg_confidence > 0.4:
            confidence_level = "보통"
        else:
            confidence_level = "낮음 (불확실)"
        
        print(f"  신뢰도 수준: {confidence_level}")
        
        return {
            'mean_prediction': float(mean_pred),
            'bias_direction': bias_direction,
            'bias_level': bias_level,
            'extreme_human_count': int(extreme_human),
            'extreme_ai_count': int(extreme_ai),
            'moderate_range_count': int(moderate_range),
            'average_confidence': float(avg_confidence),
            'confidence_level': confidence_level
        }
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return None

def suggest_improvements(analysis_results: dict):
    """분석 결과 기반 개선 방안 제안"""
    print("\n💡 개선 방안 제안")
    print("=" * 50)
    
    suggestions = []
    
    # 불균형 심각도 기반 제안
    if analysis_results.get('imbalance_ratio', 0) > 10:
        suggestions.extend([
            "🎯 Class Weight 조정 (가중치 11.2:1)",
            "🔥 Focal Loss 적용 (alpha=0.25, gamma=2.0)",
            "📊 SMOTE 오버샘플링 고려"
        ])
    
    # 편향성 기반 제안
    mean_pred = analysis_results.get('mean_prediction', 0.5)
    if mean_pred < 0.3:
        suggestions.extend([
            "⚖️ 임계값 낮추기 (0.5 → 0.3-0.4)",
            "🎯 AI 클래스 가중치 대폭 증가",
            "📈 Recall 중심 평가 지표 추가"
        ])
    elif mean_pred < 0.4:
        suggestions.extend([
            "⚖️ 임계값 조정 (0.5 → 0.4)",
            "🎯 Class Weight 미세 조정"
        ])
    
    # 신뢰도 기반 제안
    confidence = analysis_results.get('average_confidence', 0.5)
    if confidence < 0.4:
        suggestions.extend([
            "🔧 모델 복잡도 증가 (더 큰 BERT 모델)",
            "📚 데이터 증강으로 학습 안정화",
            "🎯 앙상블 가중치 재조정"
        ])
    elif confidence > 0.8:
        suggestions.extend([
            "⚠️ 과적합 확인 필요",
            "🎯 정규화 강화 (dropout, weight decay)",
            "📊 CV와 Public 점수 차이 모니터링"
        ])
    
    # 극단값 분석 기반 제안
    extreme_ratio = analysis_results.get('extreme_human_count', 0) + analysis_results.get('extreme_ai_count', 0)
    total_samples = analysis_results.get('total_samples', 1)
    
    if extreme_ratio / total_samples > 0.8:
        suggestions.append("⚠️ 과도한 확신 - Temperature Scaling 고려")
    elif extreme_ratio / total_samples < 0.2:
        suggestions.append("🎯 모델 결정력 향상 - 더 깊은 학습 필요")
    
    print("📋 우선순위별 개선 방안:")
    for i, suggestion in enumerate(suggestions[:8], 1):  # 상위 8개만
        print(f"  {i}. {suggestion}")
    
    if not suggestions:
        print("  ✅ 현재 모델이 균형적으로 잘 작동 중")
    
    return suggestions

def generate_improvement_commands(analysis_results: dict):
    """분석 결과 기반 구체적인 실행 명령어 생성"""
    print("\n🚀 즉시 실행 가능한 명령어")
    print("=" * 50)
    
    commands = []
    
    # 불균형 비율에 따른 명령어
    imbalance_ratio = analysis_results.get('imbalance_ratio', 1)
    
    if imbalance_ratio > 10:
        commands.extend([
            "# Class Weight 조정 훈련",
            "python main.py --env gpu --experiment class_weight",
            "",
            "# Focal Loss 적용 훈련", 
            "python main.py --env gpu --experiment focal_loss",
            "",
            "# 임계값 최적화",
            "python scripts/optimize_threshold.py --input submission.csv --target recall"
        ])
    
    # 편향성에 따른 명령어
    mean_pred = analysis_results.get('mean_prediction', 0.5)
    
    if mean_pred < 0.4:
        optimal_threshold = max(0.2, mean_pred - 0.1)
        commands.extend([
            "",
            f"# 임계값 조정 예측",
            f"python scripts/adjust_threshold.py --input submission.csv --threshold {optimal_threshold:.2f}"
        ])
    
    # SMOTE 명령어 (심각한 불균형 시)
    if imbalance_ratio > 8:
        commands.extend([
            "",
            "# SMOTE 오버샘플링 적용",
            "python scripts/train_with_smote.py --env gpu --sampling_strategy auto"
        ])
    
    print("```bash")
    for command in commands:
        print(command)
    print("```")
    
    return commands

def save_analysis_report(all_results: dict, output_file: str = "results/imbalance_analysis_report.json"):
    """분석 결과를 JSON 파일로 저장"""
    Path("results").mkdir(exist_ok=True)
    
    report = {
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'baseline_performance': all_results,
        'improvement_priority': [
            "class_weight_adjustment",
            "focal_loss_implementation", 
            "threshold_optimization",
            "smote_oversampling",
            "advanced_augmentation"
        ],
        'next_experiments': [
            "python main.py --env gpu --experiment class_weight",
            "python main.py --env gpu --experiment focal_loss", 
            "python scripts/optimize_threshold.py --input submission.csv"
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 상세 분석 보고서 저장: {output_file}")

def main():
    """메인 분석 함수"""
    print("🔍 클래스 불균형 영향 종합 분석")
    print("=" * 60)
    
    # 제출 파일 확인
    submission_files = ["submission.csv", "submissions/submission_*.csv"]
    submission_file = None
    
    for pattern in submission_files:
        if "*" in pattern:
            # Glob 패턴으로 최신 파일 찾기
            from glob import glob
            files = glob(pattern)
            if files:
                submission_file = max(files, key=lambda x: Path(x).stat().st_mtime)
                break
        else:
            if Path(pattern).exists():
                submission_file = pattern
                break
    
    if not submission_file:
        print("❌ 제출 파일을 찾을 수 없습니다.")
        print("   예상 위치: submission.csv 또는 submissions/submission_*.csv")
        return 1
    
    print(f"📄 분석 대상: {submission_file}")
    
    # 종합 분석 실행
    all_results = {}
    
    # 1. 예측값 분포 분석
    pred_analysis = analyze_prediction_distribution(submission_file)
    if pred_analysis:
        all_results['prediction_distribution'] = pred_analysis
    
    # 2. 클래스 불균형 영향 분석  
    imbalance_analysis = analyze_class_imbalance_impact()
    if imbalance_analysis:
        all_results['class_imbalance'] = imbalance_analysis
    
    # 3. 모델 편향성 분석
    bias_analysis = analyze_model_bias(submission_file)
    if bias_analysis:
        all_results['model_bias'] = bias_analysis
    
    # 결합된 결과로 개선 방안 제안
    combined_results = {}
    for analysis in all_results.values():
        combined_results.update(analysis)
    
    # 4. 개선 방안 제안
    suggestions = suggest_improvements(combined_results)
    all_results['improvement_suggestions'] = suggestions
    
    # 5. 실행 명령어 생성
    commands = generate_improvement_commands(combined_results)
    all_results['execution_commands'] = commands
    
    # 6. 보고서 저장
    save_analysis_report(all_results)
    
    print("\n" + "=" * 60)
    print("✅ 클래스 불균형 영향 분석 완료!")
    print("📋 다음 단계: IMBALANCE_IMPROVEMENT_PLAN.md 참조")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())