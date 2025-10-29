"""
Fast-SRGAN (MobileSRGAN) 모델 및 훈련/추론을 위한 설정 파일
"""

from dataclasses import dataclass

@dataclass
class FastSRGANConfig:
    """Fast-SRGAN 설정 클래스"""
    
    # 모델 구조 - Generator
    gen_num_features: int = 128  # 경량화를 위해 ESPCN보다 적게
    gen_num_residual_blocks: int = 8
    gen_upscale_factor: int = 1  # 1:1 복원 (업스케일링 없음)
    use_mobile_blocks: bool = True  # MobileNet-style blocks 사용
    
    # 모델 구조 - Discriminator
    disc_num_features: int = 128
    disc_num_layers: int = 6
    
    # 훈련 설정
    num_epochs: int = 10
    batch_size: int = 1
    learning_rate_gen: float = 1e-4
    learning_rate_disc: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-4
    train_split: float = 0.8
    num_workers: int = 4
    
    # 손실함수 가중치
    adversarial_weight: float = 1e-3  # GAN loss 가중치
    content_weight: float = 1.0       # Content loss 가중치
    perceptual_weight: float = 1.0    # Perceptual loss 가중치
    
    # Fast-SRGAN 특화 설정
    warmup_epochs: int = 10  # Generator만 먼저 훈련
    discriminator_update_freq: int = 1  # Discriminator 업데이트 빈도
    generator_update_freq: int = 1      # Generator 업데이트 빈도
    
    # 스케줄러
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau', 'none'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # 데이터 경로 (기존 ESPCN과 동일)
    work_dir: str = 'work_dir'
    input_channel: str = 'Y_low'
    target_channel: str = 'Y'
    output_channel: str = 'Y_fast_srgan_recon'  # Fast-SRGAN 전용 출력 폴더
    normalize: bool = True
    
    # 추론 설정
    model_path_gen: str = 'fast_srgan_generator_best.pth'
    model_path_disc: str = 'fast_srgan_discriminator_best.pth'
    use_amp: bool = True
    benchmark_frames: int = 100
    benchmark_resolution: tuple = (720, 1280)
    
    # 시스템 설정
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    
    # 검증 및 저장 설정
    save_interval: int = 1  # 몇 epoch마다 모델 저장
    validate_interval: int = 1  # 몇 epoch마다 검증
    sample_interval: int = 1  # 몇 epoch마다 샘플 이미지 저장
    
    def __post_init__(self):
        """설정 후처리"""
        if self.device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 기본 설정 인스턴스
fast_srgan_config = FastSRGANConfig()