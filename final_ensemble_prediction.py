#!/usr/bin/env python3
"""
Final ensemble prediction script that was executed inline.
Combines KoELECTRA and KLUE-BERT models with optimal weighting.
"""

import pandas as pd
import numpy as np
import torch
import json
import os
from src.config import get_config_for_environment
from src.predictor import Predictor
import logging

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load configuration for GPU environment
    config = get_config_for_environment('gpu')

    # Load test data (already in paragraph format)
    logger.info('🔄 Loading test data...')
    test_df = pd.read_csv('data/test.csv', encoding='utf-8-sig')
    logger.info(f'✅ Test data loaded: {len(test_df)} paragraphs')

    # Initialize predictor
    predictor = Predictor(config)

    # Define model paths for ensemble
    koelectra_models = [
        'models/best_model_fold_1.pt',
        'models/best_model_fold_2.pt', 
        'models/best_model_fold_3.pt'
    ]

    klue_bert_models = [
        'klue-bert/models/best_model_klue_bert_fold_1.pt',
        'klue-bert/models/best_model_klue_bert_fold_2.pt',
        'klue-bert/models/best_model_klue_bert_fold_3.pt'
    ]

    # Check model availability
    available_koelectra = [path for path in koelectra_models if os.path.exists(path)]
    available_klue_bert = [path for path in klue_bert_models if os.path.exists(path)]

    logger.info(f'📊 Available KoELECTRA models: {len(available_koelectra)}')
    logger.info(f'📊 Available KLUE-BERT models: {len(available_klue_bert)}')

    if len(available_koelectra) == 0 and len(available_klue_bert) == 0:
        logger.error('❌ No trained models found!')
        return

    # Combine all available models for ensemble
    all_models = available_koelectra + available_klue_bert
    logger.info(f'🎯 Total models for ensemble: {len(all_models)}')

    # Define weights based on performance (KLUE-BERT performs better)
    koelectra_weight = 0.4  # 40% for KoELECTRA (AUC ~0.706)
    klue_bert_weight = 0.6  # 60% for KLUE-BERT (AUC 0.7355)

    # Calculate individual model weights
    if len(available_koelectra) > 0 and len(available_klue_bert) > 0:
        # Both model types available
        koelectra_individual_weight = koelectra_weight / len(available_koelectra)
        klue_bert_individual_weight = klue_bert_weight / len(available_klue_bert)
        
        weights = ([koelectra_individual_weight] * len(available_koelectra) + 
                   [klue_bert_individual_weight] * len(available_klue_bert))
    elif len(available_klue_bert) > 0:
        # Only KLUE-BERT available
        weights = [1.0 / len(available_klue_bert)] * len(available_klue_bert)
    else:
        # Only KoELECTRA available
        weights = [1.0 / len(available_koelectra)] * len(available_koelectra)

    logger.info(f'📊 Model weights: {[f"{w:.3f}" for w in weights]}')
    logger.info(f'📊 KoELECTRA weight: {koelectra_weight}, KLUE-BERT weight: {klue_bert_weight}')

    # Perform weighted ensemble prediction
    logger.info('🚀 Starting ensemble prediction...')
    predictions = predictor.ensemble_predict(
        model_paths=all_models,
        test_df=test_df,
        method='weighted_mean',
        weights=weights
    )

    # Create submission
    logger.info('📝 Creating submission file...')
    submission_df = pd.DataFrame({
        'ID': test_df['ID'],
        'generated': predictions
    })

    # Save final ensemble submission
    os.makedirs('submissions', exist_ok=True)
    submission_path = 'submissions/final_ensemble_submission.csv'
    submission_df.to_csv(submission_path, index=False)

    # Calculate prediction statistics
    pred_array = np.array(predictions)
    stats = {
        'mean': float(np.mean(pred_array)),
        'std': float(np.std(pred_array)),
        'min': float(np.min(pred_array)),
        'max': float(np.max(pred_array)),
        'ai_ratio': float(np.sum(pred_array > 0.5) / len(pred_array) * 100)
    }

    logger.info('🎉 Final ensemble prediction completed!')
    logger.info(f'📄 Submission file: {submission_path}')
    logger.info(f'📊 Final prediction statistics:')
    logger.info(f'   평균: {stats["mean"]:.4f}')
    logger.info(f'   표준편차: {stats["std"]:.4f}')
    logger.info(f'   최소: {stats["min"]:.4f}')
    logger.info(f'   최대: {stats["max"]:.4f}')
    logger.info(f'   AI 비율(>0.5): {stats["ai_ratio"]:.2f}%')

    # Save ensemble metadata
    metadata = {
        'ensemble_type': 'weighted_mean',
        'model_paths': all_models,
        'model_weights': weights,
        'koelectra_models': len(available_koelectra),
        'klue_bert_models': len(available_klue_bert),
        'prediction_stats': stats,
        'koelectra_weight': koelectra_weight,
        'klue_bert_weight': klue_bert_weight,
        'model_performance': {
            'koelectra_auc': 0.706,
            'klue_bert_auc': 0.7355
        }
    }

    with open('submissions/ensemble_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info('✅ Ensemble metadata saved to submissions/ensemble_metadata.json')
    print('SUCCESS: Final ensemble prediction with KoELECTRA + KLUE-BERT completed')

if __name__ == "__main__":
    main()