"""
DeblurGAN-v2: 저주파에서 중~고주파 복원을 위한 설정
주파수 도메인 DC 투영 기반
"""

from dataclasses import dataclass
import torch

@dataclass
class DeblurGANConfig:
    """DeblurGAN-v2 설정"""
    
    # 모델 구조 (FPN 백본)
    num_features: int = 64              # 기본 특성맵 채널 수
    
    # 훈련 설정
    num_epochs: int = 100
    batch_size: int = 4
    learning_rate_gen: float = 1e-4
    learning_rate_disc: float = 4e-4
    
    # 손실 가중치
    lambda_pix: float = 1.0             # 픽셀 L1 손실
    lambda_perc: float = 0.1            # Perceptual 손실
    lambda_fft: float = 0.5             # 주파수 대역 가중 L1
    lambda_dc: float = 0.0              # DC 일치 (하드 투영 사용시 0)
    lambda_adv: float = 0.005           # Relativistic Adversarial 손실
    
    warmup_epochs: int = 20             # Discriminator 시작 에포크
    use_hard_dc: bool = True            # 하드 DC 투영 사용
    
    # 데이터 경로
    work_dir: str = 'work_dir'
    input_channel: str = 'Y_low'        # 저주파만 있는 입력
    target_channel: str = 'Y'           # 전체 주파수 타겟
    output_channel: str = 'Y_recon'
    
    # 시스템 설정
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    
    # 모델 저장
    model_path_gen: str = 'deblurgan_gen.pth'
    model_path_disc: str = 'deblurgan_disc.pth'
    save_interval: int = 10

config = DeblurGANConfig()
