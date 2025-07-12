#!/usr/bin/env python3
"""
KLUE-BERT 평가 스크립트

훈련된 모델의 성능을 평가하고 상세한 분석을 제공합니다.

사용법:
    python scripts/evaluate.py [--config config.yaml] [--models models/]
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config


def setup_logging(log_level: str = "INFO"):
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="KLUE-BERT 모델 성능 평가"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="설정 파일 경로"
    )
    
    parser.add_argument(
        "--results", "-r",
        type=str,
        default=None,
        help="결과 파일 경로 (cv_results.json)"
    )
    
    return parser.parse_args()


def load_results(results_path: str) -> dict:
    """결과 파일 로드"""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"결과 파일을 찾을 수 없습니다: {results_path}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_evaluation_report(results: dict):
    """평가 보고서 출력"""
    print("=" * 80)
    print("🏆 KLUE-BERT 모델 성능 평가 보고서")
    print("=" * 80)
    
    # 전체 성능
    print(f"\n📊 교차검증 성능:")
    print(f"   평균 AUC: {results['cv_auc_mean']:.4f}")
    print(f"   표준편차: {results['cv_auc_std']:.4f}")
    print(f"   95% 신뢰구간: [{results['cv_auc_mean'] - 1.96*results['cv_auc_std']:.4f}, "
          f"{results['cv_auc_mean'] + 1.96*results['cv_auc_std']:.4f}]")
    
    # 폴드별 성능
    print(f"\n📈 폴드별 AUC 점수:")
    for i, score in enumerate(results['cv_auc_scores']):
        print(f"   Fold {i+1}: {score:.4f}")
    
    # 훈련 시간
    print(f"\n⏱️ 훈련 시간:")
    print(f"   총 시간: {results['total_training_time']:.1f}초")
    print(f"   평균 폴드당: {results['total_training_time']/len(results['cv_auc_scores']):.1f}초")
    
    # 목표 대비 성과
    target_auc = 0.735
    achievement = "✅ 달성" if results['cv_auc_mean'] >= target_auc else "❌ 미달"
    print(f"\n🎯 목표 성능 (AUC 0.735) 대비: {achievement}")
    
    # 폴드별 상세 정보
    if 'fold_results' in results:
        print(f"\n📋 폴드별 상세 정보:")
        for fold_result in results['fold_results']:
            fold = fold_result['fold']
            best_metrics = fold_result['best_metrics']
            print(f"   Fold {fold}:")
            print(f"     최고 AUC: {best_metrics['val_auc']:.4f}")
            print(f"     정확도: {best_metrics['val_accuracy']:.4f}")
            print(f"     F1 점수: {best_metrics['val_f1']:.4f}")
            print(f"     훈련 시간: {fold_result['training_time']:.1f}초")


def main():
    """메인 함수"""
    args = parse_arguments()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("📊 KLUE-BERT 평가 시작")
    
    try:
        # 설정 로드
        config = load_config(args.config)
        
        # 결과 파일 경로 결정
        if args.results:
            results_path = args.results
        else:
            results_path = os.path.join(config.output.log_dir, "cv_results.json")
        
        # 결과 로드
        logger.info(f"📁 결과 파일 로딩: {results_path}")
        results = load_results(results_path)
        
        # 평가 보고서 출력
        print_evaluation_report(results)
        
        logger.info("✅ 평가 완료")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 평가 실패: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())