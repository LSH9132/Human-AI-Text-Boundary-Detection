"""
Focal Loss 구현 모듈

이 모듈은 메인 프로젝트에서 성공적으로 사용된 Focal Loss를 
독립적으로 구현한 것입니다. 클래스 불균형 문제 해결에 핵심적인 역할을 합니다.

참고: 메인 프로젝트에서 AUC 0.5 → 0.7355 돌파의 핵심 기술
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss 구현
    
    클래스 불균형 문제를 해결하기 위한 고급 손실 함수
    - alpha: 클래스 가중치 (소수 클래스에 더 높은 가중치)
    - gamma: 어려운 샘플에 집중하는 정도
    
    메인 프로젝트에서 사용된 성공적인 파라미터:
    - alpha = 0.083 (AI 클래스 비율 8.2% 기반)
    - gamma = 2.0 (표준 Focal Loss 값)
    """
    
    def __init__(self, alpha: float = 0.083, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: 클래스 가중치 (0-1 사이 값, 소수 클래스의 비율)
            gamma: focusing parameter (어려운 샘플에 집중하는 정도)
            reduction: 손실 축소 방법 ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # 로그에 사용할 정보 저장
        self.num_positive = 0
        self.num_negative = 0
        self.total_loss = 0.0
        self.batch_count = 0
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss 계산
        
        Args:
            inputs: 모델 출력 logits [batch_size, 1] 또는 [batch_size]
            targets: 실제 레이블 [batch_size] (0 또는 1)
        
        Returns:
            focal_loss: 계산된 Focal Loss
        """
        # 입력 차원 정규화
        if inputs.dim() > 1:
            inputs = inputs.squeeze(-1)
        
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
        
        # BCE loss 계산
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        
        # 확률 계산
        pt = torch.exp(-bce_loss)
        
        # Alpha 가중치 계산
        # 양성 클래스(AI)에는 alpha, 음성 클래스(Human)에는 (1-alpha)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Focal Loss 계산
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        # 통계 업데이트
        self._update_stats(targets, focal_loss)
        
        # 축소 방법에 따른 최종 손실 계산
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _update_stats(self, targets: torch.Tensor, loss: torch.Tensor):
        """통계 정보 업데이트"""
        with torch.no_grad():
            self.num_positive += targets.sum().item()
            self.num_negative += (1 - targets).sum().item()
            self.total_loss += loss.sum().item()
            self.batch_count += 1
    
    def get_stats(self) -> dict:
        """손실 통계 반환"""
        total_samples = self.num_positive + self.num_negative
        
        return {
            'positive_samples': int(self.num_positive),
            'negative_samples': int(self.num_negative),
            'total_samples': int(total_samples),
            'positive_ratio': self.num_positive / total_samples if total_samples > 0 else 0.0,
            'avg_loss': self.total_loss / total_samples if total_samples > 0 else 0.0,
            'batch_count': self.batch_count,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
    
    def reset_stats(self):
        """통계 초기화"""
        self.num_positive = 0
        self.num_negative = 0
        self.total_loss = 0.0
        self.batch_count = 0


class WeightedFocalLoss(FocalLoss):
    """
    가중치가 적용된 Focal Loss
    
    클래스별로 다른 가중치를 적용할 수 있는 확장된 버전
    """
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None, 
                 alpha: float = 0.083, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            class_weights: 클래스별 가중치 [negative_weight, positive_weight]
            alpha: Focal Loss의 alpha 파라미터
            gamma: Focal Loss의 gamma 파라미터
            reduction: 손실 축소 방법
        """
        super().__init__(alpha, gamma, reduction)
        self.class_weights = class_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """가중치가 적용된 Focal Loss 계산"""
        # 기본 Focal Loss 계산
        focal_loss = super().forward(inputs, targets)
        
        # 클래스 가중치 적용
        if self.class_weights is not None:
            if self.class_weights.device != targets.device:
                self.class_weights = self.class_weights.to(targets.device)
            
            # 각 샘플에 해당하는 가중치 선택
            weights = self.class_weights[targets.long()]
            
            if self.reduction == 'none':
                focal_loss = focal_loss * weights
            else:
                # mean이나 sum의 경우 가중평균 적용
                weighted_loss = focal_loss * weights
                if self.reduction == 'mean':
                    focal_loss = weighted_loss.sum() / weights.sum()
                else:  # sum
                    focal_loss = weighted_loss.sum()
        
        return focal_loss


def create_focal_loss(alpha: float = 0.083, gamma: float = 2.0, 
                     class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Focal Loss 생성 헬퍼 함수
    
    Args:
        alpha: 클래스 불균형 보정 파라미터
        gamma: 어려운 샘플 집중 파라미터
        class_weights: 추가 클래스 가중치 (선택사항)
    
    Returns:
        focal_loss: 설정된 Focal Loss 객체
    """
    if class_weights is not None:
        return WeightedFocalLoss(class_weights, alpha, gamma)
    else:
        return FocalLoss(alpha, gamma)


def calculate_optimal_alpha(positive_ratio: float) -> float:
    """
    데이터의 클래스 비율을 기반으로 최적 alpha 값 계산
    
    Args:
        positive_ratio: 양성 클래스(AI)의 비율 (0-1)
    
    Returns:
        optimal_alpha: 계산된 최적 alpha 값
    """
    # 일반적으로 소수 클래스의 비율을 alpha로 사용
    return positive_ratio


def test_focal_loss():
    """Focal Loss 동작 테스트"""
    print("🧪 Focal Loss 테스트 시작")
    
    # 테스트 데이터 생성
    batch_size = 100
    inputs = torch.randn(batch_size, 1)
    targets = torch.randint(0, 2, (batch_size,)).float()
    
    # 클래스 분포 확인
    positive_ratio = targets.mean().item()
    print(f"📊 테스트 데이터 양성 비율: {positive_ratio:.3f}")
    
    # Focal Loss 생성 및 테스트
    focal_loss = FocalLoss(alpha=positive_ratio, gamma=2.0)
    
    # 손실 계산
    loss = focal_loss(inputs, targets)
    print(f"💯 Focal Loss: {loss.item():.4f}")
    
    # 통계 확인
    stats = focal_loss.get_stats()
    print(f"📈 통계: {stats}")
    
    # BCE와 비교
    bce_loss = F.binary_cross_entropy_with_logits(inputs.squeeze(), targets)
    print(f"🔍 BCE Loss (참고): {bce_loss.item():.4f}")
    
    print("✅ Focal Loss 테스트 완료")


if __name__ == "__main__":
    test_focal_loss()