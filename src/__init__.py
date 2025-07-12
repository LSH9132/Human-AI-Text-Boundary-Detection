"""
KLUE-BERT AI 텍스트 탐지 프로젝트

이 패키지는 메인 프로젝트의 성공적인 KLUE-BERT 구현을 
독립적으로 재현하기 위한 전용 구현입니다.

주요 모듈:
- config: 설정 관리
- focal_loss: Focal Loss 구현
- data_processor: 데이터 처리 및 전처리
- trainer: 모델 훈련
- predictor: 예측 및 앙상블

목표 성능: AUC 0.735+
"""

__version__ = "1.0.0"
__author__ = "KLUE-BERT Project"

from .config import Config, load_config
from .focal_loss import FocalLoss
from .data_processor import KLUEDataProcessor, KLUETextDataset
from .trainer import KLUETrainer
from .predictor import KLUEPredictor

__all__ = [
    "Config",
    "load_config", 
    "FocalLoss",
    "KLUEDataProcessor",
    "KLUETextDataset",
    "KLUETrainer",
    "KLUEPredictor"
]