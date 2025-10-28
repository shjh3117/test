"""
ESPCN 모델 및 훈련/추론을 위한 간단한 설정 파일
"""

from dataclasses import dataclass

@dataclass
class ESPCNConfig:
    """ESPCN 설정 클래스"""
    
    # 모델 구조
    num_features: int = 256
    num_layers: int = 6
    kernel_size: int = 11
    activation: str = 'leaky_relu'  # 'relu', 'leaky_relu'
    use_batch_norm: bool = True
    dropout_rate: float = 0.01
    
    # 훈련 설정
    num_epochs: int = 1
    batch_size: int = 4  # T4 GPU + FP16에 최적화된 배치 크기
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    train_split: float = 0.8
    num_workers: int = 2
    
    # 손실함수 가중치
    mse_weight: float = 1.0
    l1_weight: float = 0.1
    
    # 최적화기
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'plateau'  # 'plateau', 'cosine', 'step', 'none'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # 데이터 경로
    work_dir: str = 'work_dir'
    input_channel: str = 'Y_low'
    target_channel: str = 'Y'
    output_channel: str = 'Y_low_recon'
    normalize: bool = True
    
    # 추론 설정
    model_path: str = 'espcn_model_best.pth'
    use_amp: bool = True
    benchmark_frames: int = 100
    benchmark_resolution: tuple = (720, 1280)
    
    # 시스템 설정
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    
    def __post_init__(self):
        """설정 후처리"""
        if self.device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 기본 설정 인스턴스
config = ESPCNConfig()