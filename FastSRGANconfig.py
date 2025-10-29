"""
Fast-SRGAN: 실시간 영상 화질 복원을 위한 경량화된 GAN 모델 설정

이 설정 파일은 빠른 추론 속도와 적절한 화질 복원 성능을 모두 만족하는
Fast-SRGAN 모델의 훈련 및 추론을 위한 모든 하이퍼파라미터를 정의합니다.

주요 특징:
- MobileNet 기반의 경량화된 생성자 네트워크
- 분리된 사전 훈련 전략 (Generator warmup)
- 적응적 학습률 스케줄링
- 혼합 정밀도 훈련 지원
"""

from dataclasses import dataclass

@dataclass
class FastSRGANConfig:
    """
    Fast-SRGAN 모델의 모든 설정을 담고 있는 클래스
    
    이 클래스는 다음과 같은 설정들을 포함합니다:
    - 모델 아키텍처 (생성자/판별자)
    - 훈련 하이퍼파라미터
    - 데이터 경로 및 처리 옵션
    - 추론 및 벤치마크 설정
    """
    
    # =================================================================
    # 모델 아키텍처 설정
    # =================================================================
    
    # Generator (생성자) 네트워크 구조
    gen_num_features: int = 96          # 고주파 복원을 위한 충분한 특성 맵 채널 수
    gen_num_residual_blocks: int = 12   # 더 깊은 네트워크로 복잡한 고주파 패턴 학습
    use_mobile_blocks: bool = False     # 고품질 복원을 위해 표준 conv 사용
    
    # Discriminator (판별자) 네트워크 구조  
    disc_num_features: int = 64         # 판별자 기본 특성 맵 채널 수
    disc_num_layers: int = 6            # 적절한 판별 능력을 위한 레이어 수
    
    # =================================================================
    # 훈련 하이퍼파라미터
    # =================================================================
    
    # 기본 훈련 설정
    num_epochs: int = 5                # 전체 훈련 에포크 수
    batch_size: int = 1                 # 배치 크기 (메모리 효율성을 위해 1로 설정)
    learning_rate: float = 2e-4         # 통합 학습률 (Adam 최적화기 기준)
    weight_decay: float = 1e-4          # L2 정규화 가중치 (과적합 방지)
    
    # 손실 함수 가중치 (애니메이션 고주파 복원 최적화)
    adversarial_weight: float = 5e-3    # GAN 손실 가중치 (고주파 디테일 강화)
    content_weight: float = 1.0         # MSE/L1 손실 가중치 (기본 복원 품질)
    perceptual_weight: float = 0.2      # VGG 기반 지각 손실 가중치 (애니메이션 시각적 품질)
    
    # 훈련 전략 설정
    warmup_epochs: int = 5              # Generator만 먼저 훈련하는 에포크 수 (안정적인 시작)
    
    # =================================================================
    # 데이터 처리 설정
    # =================================================================
    
    # 데이터 경로 설정
    work_dir: str = 'work_dir'                      # 작업 디렉토리 루트
    input_channel: str = 'Y_low'                    # 입력 채널 (저해상도 Y 채널)
    target_channel: str = 'Y'                       # 타겟 채널 (고해상도 Y 채널)
    output_channel: str = 'Y_fast_srgan_recon'     # 출력 저장 폴더명
    
    # 데이터 전처리
    normalize: bool = True              # 입력 정규화 활성화 ([0,1] 범위로 스케일링)
    train_split: float = 0.8           # 훈련/검증 데이터 분할 비율
    
    # =================================================================
    # 시스템 및 최적화 설정
    # =================================================================
    
    # 하드웨어 설정
    device: str = 'auto'                # 'auto': CUDA 자동 감지, 'cuda', 'cpu' 직접 지정
    num_workers: int = 4                # 데이터 로더 워커 프로세스 수 (CPU 코어 수에 따라 조정)
    use_amp: bool = True                # 혼합 정밀도 훈련 (메모리 절약 + 속도 향상)
    
    # 모델 저장 및 검증
    save_interval: int = 5              # 모델 저장 주기 (에포크 단위)
    validate_interval: int = 1          # 검증 실행 주기 (에포크 단위)
    
    # =================================================================
    # 추론 및 벤치마크 설정
    # =================================================================
    
    # 모델 파일 경로
    model_path_gen: str = 'fast_srgan_generator_best.pth'      # 훈련된 생성자 모델 경로
    model_path_disc: str = 'fast_srgan_discriminator_best.pth'  # 훈련된 판별자 모델 경로
    
    # 성능 벤치마크 설정
    benchmark_frames: int = 100         # 벤치마크에 사용할 프레임 수
    benchmark_resolution: tuple = (720, 1280)  # 벤치마크 해상도 (H, W)
    
    def __post_init__(self):
        """
        설정 초기화 후 자동으로 실행되는 후처리 함수
        
        - device 자동 설정: CUDA 사용 가능 여부에 따라 자동 선택
        - 설정 값 검증: 잘못된 설정값에 대한 기본적인 검증
        """
        # GPU 자동 감지 및 설정
        if self.device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Device auto-detected: {self.device}")
        
        # 기본적인 설정 검증
        assert 0 < self.train_split < 1, "train_split은 0과 1 사이의 값이어야 합니다"
        assert self.batch_size > 0, "batch_size는 양수여야 합니다"
        assert self.num_epochs > 0, "num_epochs는 양수여야 합니다"

# =================================================================
# 전역 설정 인스턴스
# =================================================================

# 기본 설정 인스턴스 생성
# 이 인스턴스는 다른 모듈에서 import하여 바로 사용할 수 있습니다
# 예: from FastSRGANconfig import fast_srgan_config
fast_srgan_config = FastSRGANConfig()

# 설정 요약 출력 함수
def print_config_summary():
    """현재 설정의 주요 내용을 요약하여 출력하는 함수"""
    config = fast_srgan_config
    print("=" * 60)
    print("Fast-SRGAN 설정 요약")
    print("=" * 60)
    print(f"모델 구조:")
    print(f"  - Generator 특성맵: {config.gen_num_features}, 잔차블록: {config.gen_num_residual_blocks}")
    print(f"  - Discriminator 특성맵: {config.disc_num_features}, 레이어: {config.disc_num_layers}")
    print(f"  - MobileNet 블록 사용: {config.use_mobile_blocks}")
    print(f"\n훈련 설정:")
    print(f"  - 에포크: {config.num_epochs}, 배치크기: {config.batch_size}")
    print(f"  - 학습률: {config.learning_rate}, 웜업 에포크: {config.warmup_epochs}")
    print(f"  - 혼합정밀도: {config.use_amp}, 디바이스: {config.device}")
    print(f"\n손실 가중치:")
    print(f"  - Content: {config.content_weight}, Adversarial: {config.adversarial_weight}")
    print(f"  - Perceptual: {config.perceptual_weight}")
    print("=" * 60)

# 메인 실행 시 설정 요약 출력
if __name__ == "__main__":
    print_config_summary()