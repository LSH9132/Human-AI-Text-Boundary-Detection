#!/usr/bin/env python3
"""
KLUE-BERT 훈련 스크립트

독립적인 KLUE-BERT 모델 훈련을 수행합니다.
메인 프로젝트의 AUC 0.7355 성과를 재현하기 위한 전용 스크립트입니다.

사용법:
    python scripts/train.py [--config config.yaml] [--device cuda] [--debug]
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, setup_reproducibility
from src.data_processor import KLUEDataProcessor
from src.trainer import KLUETrainer


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """로깅 설정"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="KLUE-BERT AI 텍스트 탐지 모델 훈련"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="설정 파일 경로 (기본값: config.yaml)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="사용할 디바이스 (auto, cpu, cuda, cuda:0 등)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드 활성화"
    )
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 로깅 설정
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 KLUE-BERT 훈련 시작")
    
    try:
        # 설정 로드
        config = load_config(args.config)
        
        if args.device:
            config.training.device = args.device
        
        if args.debug:
            config.training.epochs = 1
            config.training.batch_size = 4
            config.cv.n_folds = 2
        
        # 재현성 설정
        setup_reproducibility(config.seed)
        config.print_summary()
        
        # 데이터 처리
        processor = KLUEDataProcessor(config)
        train_df, _ = processor.load_data()
        processed_df = processor.preprocess_training_data(train_df)
        
        # 훈련
        trainer = KLUETrainer(config)
        cv_results = trainer.cross_validate(processor, processed_df)
        
        logger.info(f"✅ 훈련 완료! CV AUC: {cv_results['cv_auc_mean']:.4f}")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 훈련 실패: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())