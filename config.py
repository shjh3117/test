"""
전체 파이프라인 설정 파일
각 스크립트의 기본 설정을 여기서 관리합니다.
"""

import torch

# ===========================
# 공통 설정
# ===========================
WORK_DIR = 'work_dir'  # scene 데이터가 저장되는 디렉토리
INPUT_DIR = 'videos'   # 원본 비디오 디렉토리
USE_GPU = True         # False로 설정하면 CPU 사용

# GPU 실제 사용 가능 여부 확인
_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE = 'cuda' if (USE_GPU and _CUDA_AVAILABLE) else 'cpu'

# ===========================
# 01_YUV420_extractor 설정
# ===========================
class YUV420ExtractorConfig:
    # Scene 감지 설정
    SIMILARITY_THRESHOLD = 0.08     # Scene 변화 감지 임계값 (상대 변화율, 높을수록 민감)
    MIN_SCENE_FRAMES = 8            # 최소 scene 프레임 수 (8프레임 동안 새 scene 없으면 그대로 scene)
    
    # 처리 설정
    BATCH_SIZE = 240               # 한 번에 처리할 프레임 수 (GPU 메모리)
    CROP_WIDTH = 256                # 저주파 대역 크롭 너비
    CROP_HEIGHT = 144               # 저주파 대역 크롭 높이
    
    # 디바이스 설정
    DEVICE = _DEVICE


# ===========================
# 02_train 설정
# ===========================
class TrainConfig:
    DATA_ROOT = WORK_DIR            # scene별 프레임이 저장된 루트 경로
    MODEL_DIR = 'models'            # 체크포인트 저장 디렉터리
    SEED = 42                       # 재현성을 위한 시드 값
    LOG_INTERVAL = 1               # 학습 로그 출력 주기 (step)
    CHECKPOINT_INTERVAL = 1         # 에포크 단위 체크포인트 저장 주기
    EPOCHS = 10                     # 총 학습 에포크 수
    BATCH_SIZE = 1                  # 학습 배치 크기
    NUM_WORKERS = 4                 # DataLoader worker 수
    PIN_MEMORY = True               # DataLoader pin_memory 사용 여부
    BASE_CHANNELS = 96              # Generator/Discriminator 기본 채널 수
    NUM_RESIDUAL_BLOCKS = 16        # Generator 내 residual block 수 (8→16 증가)
    LR_GENERATOR = 2e-4             # Generator
    LR_DISCRIMINATOR = 2e-4         # Discriminator 학습률
    BETA1 = 0.5                     # Adam beta1
    BETA2 = 0.999                   # Adam beta2
    LAMBDA_L1 = 16                   # L1 재구성 손실 가중치
    LAMBDA_FREQ = 4                  # 저주파 보존 손실 가중치
    LAMBDA_VGG = 0.0                # VGG perceptual loss 가중치 (미사용시 0)
    SCALE_FACTOR = 5                # 업스케일 비율 (1280/256)
    USE_FP16 = True                 # FP16 Mixed Precision Training 사용 (GPU 필수)
    DEVICE = _DEVICE                # 학습 디바이스


# ===========================
# 03_recon 설정
# ===========================
class ReconConfig:
    DATA_ROOT = WORK_DIR            # 입력 scene 루트 경로
    OUTPUT_SUBDIR = 'y_recon'       # 복원 결과 저장 서브디렉터리
    CHECKPOINT_PATH = 'models/latest_generator.pt'  # Generator 가중치 경로
    BATCH_SIZE = 8                  # 추론 배치 크기
    NUM_WORKERS = 2                 # DataLoader worker 수
    PIN_MEMORY = True               # DataLoader pin_memory 사용 여부
    DEVICE = _DEVICE                # 추론 디바이스