"""
KLUE-BERT 전용 모델 훈련 모듈

이 모듈은 메인 프로젝트에서 AUC 0.7355 돌파를 달성한 
훈련 방법론을 독립적으로 구현한 것입니다.

핵심 기능:
- KLUE-BERT 모델 초기화 및 설정
- Focal Loss를 활용한 불균형 데이터 훈련
- Document-Aware 교차검증
- 혼합 정밀도 훈련
- 조기 종료 및 모델 저장
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import time
import json

from .config import Config
from .focal_loss import FocalLoss
from .data_processor import KLUEDataProcessor, KLUETextDataset


class EarlyStopping:
    """조기 종료 클래스"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001, 
                 monitor: str = 'val_auc', mode: str = 'max'):
        """
        Args:
            patience: 성능 개선이 없을 때 기다릴 에포크 수
            min_delta: 개선으로 간주할 최소 변화량
            monitor: 모니터링할 메트릭
            mode: 'max' (높을수록 좋음) 또는 'min' (낮을수록 좋음)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.best_epoch = 0
        self.best_metrics = {}
    
    def __call__(self, metrics: Dict[str, float], epoch: int) -> bool:
        """
        현재 메트릭을 기반으로 조기 종료 여부 결정
        
        Returns:
            bool: True if early stopping should be triggered
        """
        score = metrics.get(self.monitor)
        if score is None:
            return False
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_metrics = metrics.copy()
            return False
        
        # 개선 여부 확인
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            self.best_metrics = metrics.copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class KLUETrainer:
    """KLUE-BERT 전용 훈련 클래스"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 전체 설정 객체
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 디바이스 설정
        self.device = torch.device(config.training.device)
        self.logger.info(f"🔧 사용 디바이스: {self.device}")
        
        # 모델 초기화
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        
        # 훈련 상태
        self.scaler = GradScaler() if config.optimization.use_mixed_precision else None
        self.global_step = 0
        self.current_fold = 0
        
        # 결과 저장
        self.fold_results = []
        self.training_history = []
        
        self._setup_model()
        self._setup_logging()
    
    def _setup_model(self):
        """모델 및 관련 구성요소 초기화"""
        self.logger.info(f"🤖 모델 초기화: {self.config.model.name}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            use_fast=True
        )
        
        # 모델 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.name,
            num_labels=self.config.model.num_labels,
            problem_type="single_label_classification"
        )
        
        # 모델을 디바이스로 이동
        self.model.to(self.device)
        
        # 손실 함수 설정
        self.loss_fn = FocalLoss(
            alpha=self.config.focal_loss.alpha,
            gamma=self.config.focal_loss.gamma
        )
        
        self.logger.info(f"✅ 모델 로드 완료 (파라미터: {sum(p.numel() for p in self.model.parameters()):,})")
    
    def _setup_logging(self):
        """로깅 설정"""
        os.makedirs(self.config.output.log_dir, exist_ok=True)
        
        # 파일 핸들러 추가
        log_file = os.path.join(self.config.output.log_dir, "training.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def _setup_optimizer_scheduler(self, train_dataloader: DataLoader):
        """옵티마이저와 스케줄러 설정"""
        # 옵티마이저 설정
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # 스케줄러 설정
        total_steps = len(train_dataloader) * self.config.training.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=total_steps
        )
        
        self.logger.info(f"📊 총 훈련 스텝: {total_steps:,}")
        self.logger.info(f"🔥 워밍업 스텝: {self.config.training.warmup_steps:,}")
    
    def train_epoch(self, train_dataloader: DataLoader) -> Dict[str, float]:
        """한 에포크 훈련"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(train_dataloader, desc=f"Fold {self.current_fold} 훈련")
        
        for batch in pbar:
            # 배치를 디바이스로 이동
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 기울기 초기화
            self.optimizer.zero_grad()
            
            # 순전파
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch.get('token_type_ids')
                    )
                    logits = outputs.logits.squeeze(-1)
                    loss = self.loss_fn(logits, batch['labels'])
                
                # 역전파
                self.scaler.scale(loss).backward()
                
                # 기울기 클리핑
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
                
                # 옵티마이저 업데이트
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch.get('token_type_ids')
                )
                logits = outputs.logits.squeeze(-1)
                loss = self.loss_fn(logits, batch['labels'])
                
                # 역전파
                loss.backward()
                
                # 기울기 클리핑
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
                
                # 옵티마이저 업데이트
                self.optimizer.step()
            
            # 스케줄러 업데이트
            self.scheduler.step()
            
            # 통계 업데이트
            total_loss += loss.item() * len(batch['labels'])
            total_samples += len(batch['labels'])
            self.global_step += 1
            
            # 진행률 업데이트
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/total_samples:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / total_samples
        return {'train_loss': avg_loss}
    
    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """모델 평가"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_dataloader, desc=f"Fold {self.current_fold} 평가")
            
            for batch in pbar:
                # 배치를 디바이스로 이동
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 순전파
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            token_type_ids=batch.get('token_type_ids')
                        )
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch.get('token_type_ids')
                    )
                
                logits = outputs.logits.squeeze(-1)
                loss = self.loss_fn(logits, batch['labels'])
                
                # 예측값 계산
                predictions = torch.sigmoid(logits)
                
                # 결과 수집
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                
                total_loss += loss.item() * len(batch['labels'])
                total_samples += len(batch['labels'])
        
        # 메트릭 계산
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_predictions)
        
        # 임계값 0.5로 분류 메트릭 계산
        binary_predictions = (all_predictions > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, binary_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, binary_predictions, average='binary'
        )
        
        avg_loss = total_loss / total_samples
        
        return {
            'val_loss': avg_loss,
            'val_auc': auc,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        }
    
    def train_fold(self, train_dataset: KLUETextDataset, 
                   val_dataset: KLUETextDataset, fold_idx: int) -> Dict[str, Any]:
        """단일 폴드 훈련"""
        self.current_fold = fold_idx + 1
        
        self.logger.info(f"\n{'='*20} Fold {self.current_fold}/{self.config.cv.n_folds} {'='*20}")
        
        # 데이터로더 생성
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.optimization.dataloader_num_workers,
            pin_memory=self.config.optimization.pin_memory
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.prediction.batch_size,
            shuffle=False,
            num_workers=self.config.optimization.dataloader_num_workers,
            pin_memory=self.config.optimization.pin_memory
        )
        
        # 옵티마이저 및 스케줄러 설정
        self._setup_optimizer_scheduler(train_dataloader)
        
        # 조기 종료 설정
        early_stopping = EarlyStopping(
            patience=self.config.early_stopping.patience,
            min_delta=self.config.early_stopping.min_delta,
            monitor=self.config.early_stopping.monitor
        )
        
        # 모델 초기화 (매 폴드마다 새로 시작)
        self._setup_model()
        
        # 클래스 분포 로깅
        train_stats = train_dataset.get_stats()
        val_stats = val_dataset.get_stats()
        
        self.logger.info(f"📊 훈련 데이터: {train_stats['total_samples']:,} 샘플")
        self.logger.info(f"📊 검증 데이터: {val_stats['total_samples']:,} 샘플")
        self.logger.info(f"📈 훈련 AI 비율: {train_stats['positive_ratio']:.1%}")
        self.logger.info(f"📈 검증 AI 비율: {val_stats['positive_ratio']:.1%}")
        
        # 훈련 시작
        best_model_state = None
        fold_history = []
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            epoch_start = time.time()
            
            self.logger.info(f"\n--- Epoch {epoch + 1}/{self.config.training.epochs} ---")
            
            # 훈련
            train_metrics = self.train_epoch(train_dataloader)
            
            # 평가
            val_metrics = self.evaluate(val_dataloader)
            
            # 메트릭 합치기
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch + 1
            epoch_metrics['fold'] = self.current_fold
            epoch_metrics['epoch_time'] = time.time() - epoch_start
            
            fold_history.append(epoch_metrics)
            
            # 로깅
            self.logger.info(
                f"🏃 훈련 손실: {train_metrics['train_loss']:.4f} | "
                f"🎯 검증 AUC: {val_metrics['val_auc']:.4f} | "
                f"📊 검증 손실: {val_metrics['val_loss']:.4f}"
            )
            
            # 조기 종료 체크
            if early_stopping(val_metrics, epoch):
                self.logger.info(f"🛑 조기 종료 (patience={early_stopping.patience})")
                break
            
            # 최고 모델 저장
            if val_metrics['val_auc'] == early_stopping.best_score:
                best_model_state = self.model.state_dict().copy()
                
                if self.config.output.save_best_only:
                    model_path = os.path.join(
                        self.config.output.model_dir,
                        f"best_model_fold_{self.current_fold}.pt"
                    )
                    torch.save({
                        'model_state_dict': best_model_state,
                        'config': self.config.to_dict(),
                        'metrics': val_metrics,
                        'epoch': epoch + 1,
                        'fold': self.current_fold
                    }, model_path)
                    
                    self.logger.info(f"💾 최고 모델 저장: {model_path}")
        
        # 최고 성능 모델 로드
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        
        # 폴드 결과 정리
        fold_result = {
            'fold': self.current_fold,
            'best_metrics': early_stopping.best_metrics,
            'best_epoch': early_stopping.best_epoch,
            'training_time': training_time,
            'history': fold_history
        }
        
        self.logger.info(f"✅ Fold {self.current_fold} 완료 (시간: {training_time:.1f}초)")
        self.logger.info(f"🏆 최고 AUC: {early_stopping.best_metrics['val_auc']:.4f}")
        
        return fold_result
    
    def cross_validate(self, processor: KLUEDataProcessor, 
                      processed_df) -> Dict[str, Any]:
        """교차검증 훈련"""
        self.logger.info("🚀 KLUE-BERT 교차검증 훈련 시작")
        
        # Document-Aware 분할 생성
        splits = processor.create_document_aware_splits(processed_df)
        
        # 각 폴드 훈련
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            # 데이터 분할
            train_data = processed_df.iloc[train_indices]
            val_data = processed_df.iloc[val_indices]
            
            # 데이터셋 생성
            train_dataset = processor.create_dataset(
                texts=train_data['paragraph_text'].tolist(),
                labels=train_data['generated'].tolist()
            )
            
            val_dataset = processor.create_dataset(
                texts=val_data['paragraph_text'].tolist(),
                labels=val_data['generated'].tolist()
            )
            
            # 폴드 훈련
            fold_result = self.train_fold(train_dataset, val_dataset, fold_idx)
            self.fold_results.append(fold_result)
        
        # 전체 결과 정리
        cv_results = self._summarize_cv_results()
        
        # 결과 저장
        self._save_cv_results(cv_results)
        
        return cv_results
    
    def _summarize_cv_results(self) -> Dict[str, Any]:
        """교차검증 결과 요약"""
        fold_aucs = [result['best_metrics']['val_auc'] for result in self.fold_results]
        
        summary = {
            'cv_auc_mean': np.mean(fold_aucs),
            'cv_auc_std': np.std(fold_aucs),
            'cv_auc_scores': fold_aucs,
            'total_training_time': sum(result['training_time'] for result in self.fold_results),
            'config': self.config.to_dict(),
            'fold_results': self.fold_results
        }
        
        self.logger.info("\n" + "="*60)
        self.logger.info("🏁 교차검증 완료!")
        self.logger.info("="*60)
        self.logger.info(f"📊 CV AUC: {summary['cv_auc_mean']:.4f} ± {summary['cv_auc_std']:.4f}")
        self.logger.info(f"📈 폴드별 AUC: {[f'{auc:.4f}' for auc in fold_aucs]}")
        self.logger.info(f"⏱️ 총 훈련 시간: {summary['total_training_time']:.1f}초")
        
        return summary
    
    def _save_cv_results(self, cv_results: Dict[str, Any]):
        """교차검증 결과 저장"""
        results_path = os.path.join(self.config.output.log_dir, "cv_results.json")
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(cv_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 결과 저장: {results_path}")


def test_trainer():
    """훈련기 테스트"""
    from .config import Config
    
    print("🧪 KLUE 훈련기 테스트 시작")
    
    # 설정 로드
    config = Config()
    config.training.epochs = 1  # 테스트용
    config.training.batch_size = 4
    
    # 훈련기 생성
    trainer = KLUETrainer(config)
    
    print(f"🤖 모델: {trainer.model.__class__.__name__}")
    print(f"🔧 디바이스: {trainer.device}")
    print(f"💾 파라미터 수: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    print("✅ 훈련기 테스트 완료")


if __name__ == "__main__":
    test_trainer()