#!/usr/bin/env python3
"""
KLUE-BERT 예측 스크립트

훈련된 KLUE-BERT 모델을 사용하여 테스트 데이터에 대한 예측을 생성합니다.

사용법:
    python scripts/predict.py [--config config.yaml] [--models models/] [--output submission.csv]
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data_processor import KLUEDataProcessor
from src.predictor import KLUEPredictor


def setup_logging(log_level: str = "INFO"):
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="KLUE-BERT AI 텍스트 탐지 예측"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="설정 파일 경로"
    )
    
    parser.add_argument(
        "--models", "-m",
        type=str,
        default=None,
        help="모델 디렉토리 경로"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="출력 파일 경로"
    )
    
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="테스트 데이터 파일 경로"
    )
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_arguments()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("🔮 KLUE-BERT 예측 시작")
    
    try:
        # 설정 로드
        config = load_config(args.config)
        
        if args.test_data:
            config.data.test_file = args.test_data
        
        if args.output:
            config.output.submission_file = args.output
        
        # 데이터 처리기 생성
        processor = KLUEDataProcessor(config)
        _, test_df = processor.load_data()
        
        # 예측기 생성
        predictor = KLUEPredictor(config)
        
        # 모델 로드
        model_dir = args.models if args.models else config.output.model_dir
        num_models = predictor.load_models(model_dir)
        logger.info(f"📥 로드된 모델 수: {num_models}")
        
        # 예측 수행
        predictions_df = predictor.predict_test_data(processor, test_df)
        
        # 결과 저장
        output_file = predictor.save_predictions(predictions_df)
        
        # 검증
        is_valid = predictor.validate_predictions(predictions_df)
        
        if is_valid:
            logger.info(f"🎉 예측 완료: {output_file}")
            return 0
        else:
            logger.error("❌ 예측 결과 검증 실패")
            return 1
            
    except Exception as e:
        logger.error(f"❌ 예측 실패: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())