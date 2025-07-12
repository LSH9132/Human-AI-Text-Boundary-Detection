"""
KLUE-BERT 전용 데이터 처리 모듈

이 모듈은 메인 프로젝트의 성공적인 데이터 처리 방식을 
KLUE-BERT에 특화하여 독립적으로 구현한 것입니다.

핵심 기능:
- 문서 단위 단락 분할
- 한국어 텍스트 전처리
- Document-Aware 데이터 분할
- KLUE-BERT 토크나이저 최적화
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import logging
from sklearn.model_selection import StratifiedKFold, GroupKFold

from .config import Config


class KLUETextDataset(Dataset):
    """KLUE-BERT 전용 데이터셋 클래스"""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, 
                 tokenizer: AutoTokenizer = None, max_length: int = 512):
        """
        Args:
            texts: 입력 텍스트 리스트
            labels: 레이블 리스트 (0: Human, 1: AI)
            tokenizer: KLUE-BERT 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 통계 정보
        self.total_samples = len(texts)
        if labels is not None:
            self.positive_samples = sum(labels)
            self.negative_samples = self.total_samples - self.positive_samples
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """배치 아이템 반환"""
        text = str(self.texts[idx])
        
        # KLUE-BERT 토크나이징
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return item
    
    def get_stats(self) -> Dict[str, Any]:
        """데이터셋 통계 반환"""
        stats = {
            'total_samples': self.total_samples,
            'avg_text_length': np.mean([len(text) for text in self.texts])
        }
        
        if self.labels is not None:
            stats.update({
                'positive_samples': self.positive_samples,
                'negative_samples': self.negative_samples,
                'positive_ratio': self.positive_samples / self.total_samples,
                'class_balance_ratio': f"{self.negative_samples}:{self.positive_samples}"
            })
        
        return stats


class KLUEDataProcessor:
    """KLUE-BERT 전용 데이터 처리 클래스"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 전체 설정 객체
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.name,
            use_fast=True
        )
        
        # 통계 정보 저장
        self.stats = {}
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """훈련 및 테스트 데이터 로드"""
        self.logger.info("📊 데이터 파일 로딩 중...")
        
        try:
            # 훈련 데이터 로드
            train_df = pd.read_csv(self.config.data.train_file)
            test_df = pd.read_csv(self.config.data.test_file)
            
            self.logger.info(f"✅ 훈련 데이터: {len(train_df):,} 샘플")
            self.logger.info(f"✅ 테스트 데이터: {len(test_df):,} 샘플")
            
            # 데이터 검증
            self._validate_data(train_df, test_df)
            
            return train_df, test_df
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 로딩 실패: {e}")
            raise
    
    def _validate_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """데이터 무결성 검증"""
        # 필수 컬럼 확인
        required_train_cols = ['title', 'full_text', 'generated']
        required_test_cols = ['title', 'full_text']
        
        missing_train = [col for col in required_train_cols if col not in train_df.columns]
        missing_test = [col for col in required_test_cols if col not in test_df.columns]
        
        if missing_train:
            raise ValueError(f"훈련 데이터에 필수 컬럼이 없습니다: {missing_train}")
        if missing_test:
            raise ValueError(f"테스트 데이터에 필수 컬럼이 없습니다: {missing_test}")
        
        # 레이블 분포 확인
        label_dist = train_df['generated'].value_counts()
        self.logger.info(f"📈 레이블 분포: {dict(label_dist)}")
        
        # 클래스 불균형 확인
        positive_ratio = train_df['generated'].mean()
        self.logger.info(f"📊 AI 클래스 비율: {positive_ratio:.1%}")
        
        if positive_ratio < 0.05 or positive_ratio > 0.95:
            self.logger.warning("⚠️ 심각한 클래스 불균형이 감지되었습니다!")
    
    def clean_korean_text(self, text: str) -> str:
        """한국어 텍스트 정제"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).strip()
        
        # 특수 문자 정제 (한국어 친화적)
        text = re.sub(r'[^\w\s가-힣.,!?;:\'"()[\]{}/-]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 연속된 구두점 정리
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def split_into_paragraphs(self, text: str, title: str = "") -> List[str]:
        """텍스트를 단락으로 분할 (한국어 최적화)"""
        if pd.isna(text) or not text:
            return []
        
        # 텍스트 정제
        text = self.clean_korean_text(text)
        
        # 단락 분할 (한국어 문장 구분자 고려)
        paragraphs = []
        
        # 줄바꿈 기준 분할
        lines = text.split('\n')
        
        current_paragraph = ""
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                    current_paragraph = ""
                continue
            
            # 문장이 너무 짧으면 합치기
            if len(current_paragraph) < 50:
                current_paragraph += " " + line if current_paragraph else line
            else:
                paragraphs.append(current_paragraph)
                current_paragraph = line
        
        # 마지막 단락 추가
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        # 단락 필터링
        filtered_paragraphs = []
        for p in paragraphs:
            # 최소 길이 체크
            if len(p.strip()) < 20:
                continue
            
            # 최대 토큰 길이 체크 (대략적)
            if len(p) > self.config.model.max_length * 4:  # 한국어는 평균 4글자/토큰
                # 긴 단락은 문장 단위로 재분할
                sentences = re.split(r'[.!?]\s+', p)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk + sent) < self.config.model.max_length * 3:
                        current_chunk += sent + ". "
                    else:
                        if current_chunk.strip():
                            filtered_paragraphs.append(current_chunk.strip())
                        current_chunk = sent + ". "
                if current_chunk.strip():
                    filtered_paragraphs.append(current_chunk.strip())
            else:
                filtered_paragraphs.append(p.strip())
        
        # 최대 단락 수 제한
        if len(filtered_paragraphs) > self.config.data.max_paragraphs_per_doc:
            # 길이 기준으로 정렬하여 가장 긴 단락들 선택
            filtered_paragraphs.sort(key=len, reverse=True)
            filtered_paragraphs = filtered_paragraphs[:self.config.data.max_paragraphs_per_doc]
        
        return filtered_paragraphs
    
    def preprocess_training_data(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """훈련 데이터 전처리 - 단락 분할"""
        self.logger.info("🔄 훈련 데이터 전처리 중...")
        
        processed_data = []
        total_docs = len(train_df)
        
        for idx, row in train_df.iterrows():
            if idx % 10000 == 0:
                self.logger.info(f"처리 진행률: {idx:,}/{total_docs:,}")
            
            title = str(row['title'])
            full_text = str(row['full_text'])
            label = int(row['generated'])
            
            # 단락 분할
            paragraphs = self.split_into_paragraphs(full_text, title)
            
            # 각 단락을 개별 샘플로 추가
            for para_idx, paragraph in enumerate(paragraphs):
                processed_data.append({
                    'title': title,
                    'paragraph_text': paragraph,
                    'paragraph_idx': para_idx,
                    'original_idx': idx,
                    'generated': label
                })
        
        processed_df = pd.DataFrame(processed_data)
        
        # 통계 업데이트
        self.stats['original_documents'] = total_docs
        self.stats['processed_paragraphs'] = len(processed_df)
        self.stats['avg_paragraphs_per_doc'] = len(processed_df) / total_docs
        
        self.logger.info(f"✅ 전처리 완료: {total_docs:,} 문서 → {len(processed_df):,} 단락")
        self.logger.info(f"📊 문서당 평균 단락 수: {self.stats['avg_paragraphs_per_doc']:.1f}")
        
        return processed_df
    
    def create_document_aware_splits(self, processed_df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Document-Aware 교차검증 분할 생성"""
        self.logger.info("📊 Document-Aware 교차검증 분할 생성 중...")
        
        # 문서별 대표 레이블 계산 (다수결)
        doc_labels = processed_df.groupby('title')['generated'].agg(lambda x: x.mode().iloc[0])
        
        # GroupKFold를 사용하여 문서 단위 분할
        group_kfold = GroupKFold(n_splits=self.config.cv.n_folds)
        
        splits = []
        for fold_idx, (train_docs, val_docs) in enumerate(group_kfold.split(
            X=doc_labels.index,
            y=doc_labels.values,
            groups=doc_labels.index
        )):
            # 문서 인덱스를 단락 인덱스로 변환
            train_titles = doc_labels.index[train_docs]
            val_titles = doc_labels.index[val_docs]
            
            train_indices = processed_df[processed_df['title'].isin(train_titles)].index.values
            val_indices = processed_df[processed_df['title'].isin(val_titles)].index.values
            
            splits.append((train_indices, val_indices))
            
            # 분할 통계
            train_pos = processed_df.iloc[train_indices]['generated'].sum()
            val_pos = processed_df.iloc[val_indices]['generated'].sum()
            train_total = len(train_indices)
            val_total = len(val_indices)
            
            self.logger.info(
                f"Fold {fold_idx + 1}: "
                f"훈련 {train_total:,}개 (AI: {train_pos:,}, {train_pos/train_total:.1%}) | "
                f"검증 {val_total:,}개 (AI: {val_pos:,}, {val_pos/val_total:.1%})"
            )
        
        return splits
    
    def create_dataset(self, texts: List[str], labels: Optional[List[int]] = None) -> KLUETextDataset:
        """KLUE 데이터셋 생성"""
        return KLUETextDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.config.model.max_length
        )
    
    def get_class_weights(self, labels: List[int]) -> torch.Tensor:
        """클래스 가중치 계산"""
        label_counts = np.bincount(labels)
        total = len(labels)
        
        # 역빈도 가중치
        weights = total / (len(label_counts) * label_counts)
        
        return torch.FloatTensor(weights)
    
    def analyze_text_stats(self, texts: List[str]) -> Dict[str, float]:
        """텍스트 통계 분석"""
        if not texts:
            return {}
        
        lengths = [len(text) for text in texts]
        token_counts = [len(self.tokenizer.encode(text, add_special_tokens=False)) for text in texts[:1000]]  # 샘플링
        
        return {
            'avg_char_length': np.mean(lengths),
            'median_char_length': np.median(lengths),
            'max_char_length': np.max(lengths),
            'avg_token_count': np.mean(token_counts),
            'max_token_count': np.max(token_counts),
            'texts_over_max_length': sum(1 for tc in token_counts if tc > self.config.model.max_length)
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """데이터 처리 통계 반환"""
        return self.stats.copy()


def test_data_processor():
    """데이터 처리 테스트"""
    from .config import Config
    
    print("🧪 KLUE 데이터 처리 테스트 시작")
    
    # 설정 로드
    config = Config()
    
    # 데이터 처리기 생성
    processor = KLUEDataProcessor(config)
    
    # 테스트 텍스트
    test_text = """
    안녕하세요. 이것은 테스트 텍스트입니다.
    
    한국어 텍스트 처리를 테스트하고 있습니다. 
    KLUE-BERT 모델을 사용하여 AI 생성 텍스트를 탐지합니다.
    
    이 시스템은 문서를 단락으로 분할하고, 각 단락을 독립적으로 분류합니다.
    """
    
    # 단락 분할 테스트
    paragraphs = processor.split_into_paragraphs(test_text)
    print(f"📄 분할된 단락 수: {len(paragraphs)}")
    for i, para in enumerate(paragraphs):
        print(f"  {i+1}: {para[:50]}...")
    
    # 데이터셋 생성 테스트
    dataset = processor.create_dataset(paragraphs, [0, 1, 0])
    print(f"📊 데이터셋 크기: {len(dataset)}")
    
    # 샘플 아이템 확인
    sample = dataset[0]
    print(f"🔍 샘플 형태: {list(sample.keys())}")
    print(f"   input_ids: {sample['input_ids'].shape}")
    print(f"   attention_mask: {sample['attention_mask'].shape}")
    
    print("✅ 데이터 처리 테스트 완료")


if __name__ == "__main__":
    test_data_processor()