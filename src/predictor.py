"""
KLUE-BERT 전용 예측 시스템

이 모듈은 메인 프로젝트에서 성공적으로 구현된 예측 시스템을 
독립적으로 구현한 것입니다.

핵심 기능:
- 앙상블 예측 (다중 폴드 모델 결합)
- 배치 단위 효율적 처리
- 메모리 최적화
- 결과 후처리 및 검증
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from tqdm import tqdm
import time
import json

from .config import Config
from .data_processor import KLUEDataProcessor, KLUETextDataset
from transformers import AutoModelForSequenceClassification


class KLUEPredictor:
    """KLUE-BERT 전용 예측 클래스"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 전체 설정 객체
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 디바이스 설정
        self.device = torch.device(config.training.device)
        self.logger.info(f"🔧 예측 디바이스: {self.device}")
        
        # 모델 관련
        self.models = []
        self.model_paths = []
        self.ensemble_weights = None
        
        # 성능 최적화
        self.use_mixed_precision = config.optimization.use_mixed_precision
        
        # 통계
        self.prediction_stats = {}
    
    def load_models(self, model_dir: Optional[str] = None) -> int:
        """훈련된 모델들 로드"""
        if model_dir is None:
            model_dir = self.config.output.model_dir
        
        self.logger.info(f"📁 모델 디렉토리: {model_dir}")
        
        # 모델 파일 찾기
        model_files = []
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith("best_model_fold_") and file.endswith(".pt"):
                    model_files.append(os.path.join(model_dir, file))
        
        model_files.sort()  # 폴드 순서대로 정렬
        
        if not model_files:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_dir}")
        
        self.logger.info(f"🔍 발견된 모델: {len(model_files)}개")
        
        # 각 모델 로드
        for model_path in model_files:
            try:
                self.logger.info(f"📥 모델 로딩: {os.path.basename(model_path)}")
                
                # 체크포인트 로드
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 모델 생성
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model.name,
                    num_labels=self.config.model.num_labels,
                    problem_type="single_label_classification"
                )
                
                # 가중치 로드
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                self.models.append(model)
                self.model_paths.append(model_path)
                
                # 모델 성능 정보 로깅
                if 'metrics' in checkpoint:
                    metrics = checkpoint['metrics']
                    self.logger.info(f"   📊 검증 AUC: {metrics.get('val_auc', 'N/A'):.4f}")
                
            except Exception as e:
                self.logger.error(f"❌ 모델 로딩 실패 {model_path}: {e}")
                continue
        
        if not self.models:
            raise RuntimeError("사용 가능한 모델이 없습니다.")
        
        self.logger.info(f"✅ 총 {len(self.models)}개 모델 로드 완료")
        return len(self.models)
    
    def set_ensemble_weights(self, weights: Optional[List[float]] = None):
        """앙상블 가중치 설정"""
        if weights is None:
            # 균등 가중치 사용
            weights = [1.0 / len(self.models)] * len(self.models)
        
        if len(weights) != len(self.models):
            raise ValueError(f"가중치 개수({len(weights)})와 모델 개수({len(self.models)})가 일치하지 않습니다.")
        
        # 정규화
        total_weight = sum(weights)
        self.ensemble_weights = [w / total_weight for w in weights]
        
        self.logger.info(f"⚖️ 앙상블 가중치: {[f'{w:.3f}' for w in self.ensemble_weights]}")
    
    def predict_batch(self, dataset: KLUETextDataset) -> np.ndarray:
        """배치 단위 예측"""
        if not self.models:
            raise RuntimeError("로드된 모델이 없습니다. load_models()를 먼저 호출하세요.")
        
        # 앙상블 가중치 설정 (아직 설정되지 않은 경우)
        if self.ensemble_weights is None:
            self.set_ensemble_weights()
        
        # 데이터로더 생성
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.prediction.batch_size,
            shuffle=False,
            num_workers=self.config.optimization.dataloader_num_workers,
            pin_memory=self.config.optimization.pin_memory
        )
        
        self.logger.info(f"🎯 예측 시작 (배치 크기: {self.config.prediction.batch_size})")
        
        all_predictions = []
        total_samples = 0
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="예측 진행")):
                # 배치를 디바이스로 이동
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                
                batch_predictions = []
                
                # 각 모델에서 예측
                for model_idx, model in enumerate(self.models):
                    if self.use_mixed_precision:
                        with autocast():
                            outputs = model(
                                input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                token_type_ids=batch.get('token_type_ids')
                            )
                    else:
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            token_type_ids=batch.get('token_type_ids')
                        )
                    
                    # 시그모이드 적용하여 확률로 변환
                    logits = outputs.logits.squeeze(-1)
                    probs = torch.sigmoid(logits)
                    
                    # 가중치 적용
                    weighted_probs = probs * self.ensemble_weights[model_idx]
                    batch_predictions.append(weighted_probs.cpu().numpy())
                
                # 앙상블 결합
                if self.config.prediction.ensemble_method == "average":
                    ensemble_pred = np.sum(batch_predictions, axis=0)
                elif self.config.prediction.ensemble_method == "weighted":
                    ensemble_pred = np.sum(batch_predictions, axis=0)
                else:
                    raise ValueError(f"지원하지 않는 앙상블 방법: {self.config.prediction.ensemble_method}")
                
                all_predictions.extend(ensemble_pred)
                total_samples += len(ensemble_pred)
        
        prediction_time = time.time() - start_time
        
        # 통계 업데이트
        self.prediction_stats = {
            'total_samples': total_samples,
            'prediction_time': prediction_time,
            'samples_per_second': total_samples / prediction_time,
            'num_models': len(self.models),
            'batch_size': self.config.prediction.batch_size
        }
        
        self.logger.info(f"✅ 예측 완료: {total_samples:,} 샘플 ({prediction_time:.1f}초)")
        self.logger.info(f"📈 처리 속도: {self.prediction_stats['samples_per_second']:.1f} 샘플/초")
        
        return np.array(all_predictions)
    
    def predict_test_data(self, processor: KLUEDataProcessor, 
                         test_df: pd.DataFrame) -> pd.DataFrame:
        """테스트 데이터 예측"""
        self.logger.info("🔮 테스트 데이터 예측 시작")
        
        # 테스트 데이터 전처리
        test_texts = []
        test_ids = []
        
        for idx, row in test_df.iterrows():
            # 단락 분할
            paragraphs = processor.split_into_paragraphs(
                row['full_text'], 
                row['title']
            )
            
            if not paragraphs:
                # 빈 텍스트인 경우 원본 사용
                paragraphs = [str(row['full_text'])]
            
            # 각 단락 추가
            for para in paragraphs:
                test_texts.append(para)
                test_ids.append(row['ID'] if 'ID' in row else f"TEST_{idx:04d}")
        
        self.logger.info(f"📊 테스트 단락 수: {len(test_texts):,}")
        
        # 데이터셋 생성
        test_dataset = processor.create_dataset(test_texts)
        
        # 예측 수행
        predictions = self.predict_batch(test_dataset)
        
        # 결과 정리
        results_df = pd.DataFrame({
            'ID': test_ids,
            'paragraph_text': test_texts,
            'prediction': predictions
        })
        
        # 문서별 예측 집계 (평균)
        if len(set(test_ids)) < len(test_ids):  # 여러 단락이 있는 경우
            self.logger.info("📊 문서별 예측 집계 중...")
            
            final_predictions = []
            unique_ids = test_df['ID'].tolist() if 'ID' in test_df.columns else [f"TEST_{i:04d}" for i in range(len(test_df))]
            
            for test_id in unique_ids:
                doc_predictions = results_df[results_df['ID'] == test_id]['prediction'].values
                if len(doc_predictions) > 0:
                    # 문서 내 단락들의 평균 예측값
                    final_pred = np.mean(doc_predictions)
                else:
                    final_pred = 0.5  # 기본값
                
                final_predictions.append(final_pred)
            
            # 최종 결과 생성
            submission_df = pd.DataFrame({
                'ID': unique_ids,
                'generated': final_predictions
            })
        else:
            # 단락별 예측을 그대로 사용
            submission_df = pd.DataFrame({
                'ID': test_ids,
                'generated': predictions
            })
        
        self.logger.info(f"📝 최종 예측 수: {len(submission_df):,}")
        
        return submission_df
    
    def save_predictions(self, predictions_df: pd.DataFrame, 
                        output_path: Optional[str] = None) -> str:
        """예측 결과 저장"""
        if output_path is None:
            output_path = self.config.output.submission_file
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # CSV 저장
        predictions_df.to_csv(output_path, index=False)
        
        self.logger.info(f"💾 예측 결과 저장: {output_path}")
        
        # 통계 로깅
        pred_stats = self._analyze_predictions(predictions_df['generated'].values)
        self.logger.info(f"📊 예측 통계: {pred_stats}")
        
        return output_path
    
    def _analyze_predictions(self, predictions: np.ndarray) -> Dict[str, Any]:
        """예측 결과 통계 분석"""
        return {
            'count': len(predictions),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'q25': float(np.percentile(predictions, 25)),
            'median': float(np.median(predictions)),
            'q75': float(np.percentile(predictions, 75)),
            'ai_ratio_05': float(np.mean(predictions > 0.5)),
            'ai_ratio_03': float(np.mean(predictions > 0.3)),
            'ai_ratio_07': float(np.mean(predictions > 0.7))
        }
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """예측 통계 반환"""
        return self.prediction_stats.copy()
    
    def validate_predictions(self, predictions_df: pd.DataFrame, 
                           sample_submission_path: Optional[str] = None) -> bool:
        """예측 결과 검증"""
        try:
            # 필수 컬럼 확인
            required_cols = ['ID', 'generated']
            missing_cols = [col for col in required_cols if col not in predictions_df.columns]
            
            if missing_cols:
                self.logger.error(f"❌ 필수 컬럼 누락: {missing_cols}")
                return False
            
            # 예측값 범위 확인
            predictions = predictions_df['generated'].values
            if np.any(predictions < 0) or np.any(predictions > 1):
                self.logger.error("❌ 예측값이 [0, 1] 범위를 벗어났습니다.")
                return False
            
            # NaN 확인
            if predictions_df.isnull().sum().sum() > 0:
                self.logger.error("❌ NaN 값이 포함되어 있습니다.")
                return False
            
            # 샘플 제출 파일과 비교
            if sample_submission_path and os.path.exists(sample_submission_path):
                sample_df = pd.read_csv(sample_submission_path)
                
                if len(predictions_df) != len(sample_df):
                    self.logger.error(f"❌ 예측 수가 다릅니다: {len(predictions_df)} vs {len(sample_df)}")
                    return False
                
                # ID 순서 확인
                if not predictions_df['ID'].equals(sample_df['ID']):
                    self.logger.warning("⚠️ ID 순서가 다릅니다. 정렬을 권장합니다.")
            
            self.logger.info("✅ 예측 결과 검증 통과")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 검증 중 오류: {e}")
            return False


def create_submission(config_path: str = "config.yaml", 
                     test_data_path: Optional[str] = None,
                     output_path: Optional[str] = None) -> str:
    """제출 파일 생성 헬퍼 함수"""
    from .config import load_config
    
    # 설정 로드
    config = load_config(config_path)
    
    # 테스트 데이터 경로 설정
    if test_data_path:
        config.data.test_file = test_data_path
    
    # 출력 경로 설정
    if output_path:
        config.output.submission_file = output_path
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 KLUE-BERT 예측 시작")
    
    # 데이터 처리기 생성
    processor = KLUEDataProcessor(config)
    
    # 테스트 데이터 로드
    _, test_df = processor.load_data()
    
    # 예측기 생성
    predictor = KLUEPredictor(config)
    
    # 모델 로드
    predictor.load_models()
    
    # 예측 수행
    predictions_df = predictor.predict_test_data(processor, test_df)
    
    # 결과 저장
    output_file = predictor.save_predictions(predictions_df)
    
    # 검증
    predictor.validate_predictions(predictions_df, config.data.submission_file)
    
    logger.info(f"🎉 예측 완료: {output_file}")
    return output_file


def test_predictor():
    """예측기 테스트"""
    from .config import Config
    
    print("🧪 KLUE 예측기 테스트 시작")
    
    # 설정 로드
    config = Config()
    config.prediction.batch_size = 4
    
    # 예측기 생성
    predictor = KLUEPredictor(config)
    
    print(f"🔧 디바이스: {predictor.device}")
    print(f"⚖️ 앙상블 방법: {config.prediction.ensemble_method}")
    
    print("✅ 예측기 테스트 완료")


if __name__ == "__main__":
    test_predictor()