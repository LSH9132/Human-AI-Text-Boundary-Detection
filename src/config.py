"""
KLUE-BERT 프로젝트 설정 관리 모듈

이 모듈은 메인 프로젝트의 성공적인 KLUE-BERT 구현을 참고하여
독립적으로 구현된 설정 관리 시스템입니다.
"""

import os
import yaml
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """데이터 관련 설정"""
    train_file: str = "data/train.csv"
    test_file: str = "data/test.csv"
    submission_file: str = "data/sample_submission.csv"
    max_paragraphs_per_doc: int = 15


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    name: str = "klue/bert-base"
    num_labels: int = 1
    max_length: int = 512


@dataclass
class TrainingConfig:
    """훈련 관련 설정"""
    device: str = "auto"
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


@dataclass
class CrossValidationConfig:
    """교차검증 관련 설정"""
    n_folds: int = 3
    strategy: str = "document_aware"
    shuffle: bool = True


@dataclass
class FocalLossConfig:
    """Focal Loss 관련 설정"""
    alpha: float = 0.083
    gamma: float = 2.0


@dataclass
class EarlyStoppingConfig:
    """조기 종료 관련 설정"""
    patience: int = 3
    min_delta: float = 0.001
    monitor: str = "val_auc"


@dataclass
class OutputConfig:
    """출력 관련 설정"""
    model_dir: str = "models"
    log_dir: str = "logs"
    submission_file: str = "submission.csv"
    save_best_only: bool = True


@dataclass
class OptimizationConfig:
    """성능 최적화 관련 설정"""
    use_mixed_precision: bool = True
    dataloader_num_workers: int = 4
    pin_memory: bool = True


@dataclass
class PredictionConfig:
    """예측 관련 설정"""
    batch_size: int = 32
    ensemble_method: str = "average"
    temperature_scaling: bool = False


@dataclass
class Config:
    """전체 설정을 관리하는 메인 클래스"""
    
    # 실험 정보
    experiment_name: str = "klue_bert_detection"
    experiment_version: str = "v1.0"
    seed: int = 42
    
    # 각 구성요소별 설정
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    focal_loss: FocalLossConfig = field(default_factory=FocalLossConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    
    def __post_init__(self):
        """초기화 후 처리"""
        # 디바이스 자동 설정
        if self.training.device == "auto":
            self.training.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 디렉토리 생성
        os.makedirs(self.output.model_dir, exist_ok=True)
        os.makedirs(self.output.log_dir, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """YAML 파일에서 설정 로드"""
        if not os.path.exists(config_path):
            print(f"설정 파일을 찾을 수 없습니다: {config_path}")
            print("기본 설정을 사용합니다.")
            return cls()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """딕셔너리에서 설정 생성"""
        config = cls()
        
        # 실험 정보 업데이트
        if 'experiment' in config_dict:
            exp_config = config_dict['experiment']
            config.experiment_name = exp_config.get('name', config.experiment_name)
            config.experiment_version = exp_config.get('version', config.experiment_version)
            config.seed = exp_config.get('seed', config.seed)
        
        # 각 섹션별 설정 업데이트
        section_mapping = {
            'data': DataConfig,
            'model': ModelConfig,
            'training': TrainingConfig,
            'cross_validation': CrossValidationConfig,
            'focal_loss': FocalLossConfig,
            'early_stopping': EarlyStoppingConfig,
            'output': OutputConfig,
            'optimization': OptimizationConfig,
            'prediction': PredictionConfig
        }
        
        for section_name, section_class in section_mapping.items():
            if section_name in config_dict:
                section_dict = config_dict[section_name]
                current_section = getattr(config, section_name.replace('cross_validation', 'cv'))
                
                # 기존 설정 업데이트
                for key, value in section_dict.items():
                    if hasattr(current_section, key):
                        setattr(current_section, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'experiment': {
                'name': self.experiment_name,
                'version': self.experiment_version,
                'seed': self.seed
            },
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'cross_validation': self.cv.__dict__,
            'focal_loss': self.focal_loss.__dict__,
            'early_stopping': self.early_stopping.__dict__,
            'output': self.output.__dict__,
            'optimization': self.optimization.__dict__,
            'prediction': self.prediction.__dict__
        }
    
    def save_yaml(self, save_path: str):
        """설정을 YAML 파일로 저장"""
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def get_experiment_id(self) -> str:
        """실험 ID 생성"""
        return f"{self.experiment_name}_{self.experiment_version}"
    
    def print_summary(self):
        """설정 요약 출력"""
        print("=" * 60)
        print(f"🚀 KLUE-BERT AI 텍스트 탐지 설정")
        print("=" * 60)
        print(f"실험명: {self.experiment_name}")
        print(f"버전: {self.experiment_version}")
        print(f"시드: {self.seed}")
        print(f"모델: {self.model.name}")
        print(f"디바이스: {self.training.device}")
        print(f"배치 크기: {self.training.batch_size}")
        print(f"최대 길이: {self.model.max_length}")
        print(f"학습률: {self.training.learning_rate}")
        print(f"에포크: {self.training.epochs}")
        print(f"교차검증: {self.cv.n_folds}-fold {self.cv.strategy}")
        print(f"Focal Loss: α={self.focal_loss.alpha}, γ={self.focal_loss.gamma}")
        print("=" * 60)


def load_config(config_path: Optional[str] = None) -> Config:
    """설정 로드 헬퍼 함수"""
    if config_path is None:
        config_path = "config.yaml"
    
    return Config.from_yaml(config_path)


def setup_reproducibility(seed: int = 42):
    """재현성을 위한 시드 설정"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # 설정 테스트
    config = load_config()
    config.print_summary()