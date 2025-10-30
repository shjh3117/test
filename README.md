# 주파수 공간 기반 Scene 검출 및 GAN 학습 파이프라인

## 📋 개요

이 프로젝트는 주파수 공간에서 scene을 검출하고, 저주파 정보만으로 고해상도 이미지를 생성하는 GAN 모델을 학습합니다.

### 핵심 아이디어
- **주파수 공간 scene 검출**: FFT 변환 후 주파수 공간에서 유사도를 비교하여 더 정확한 scene 분리
- **저주파 → 고해상도**: 256x144 저주파 이미지를 1280x720 고해상도로 복원
- **Y 채널 집중**: 휘도(Y) 정보만 처리하여 효율성 향상

## 🚀 파이프라인

### 1. Scene 검출 및 프레임 추출 (`01_YUV420_extractor.py`)

**새로운 방식:**
1. **2400개 Y 프레임**을 GPU에 로드
2. **FFT & Shift** 수행
3. **주파수 공간**에서 유사도로 Scene 검출
   - 240프레임 동안 새 scene이 없으면 → 그대로 하나의 scene
4. Scene의 **중앙 256x144 패치**를 크롭 (저주파 대역)
5. **역 Shift & 역 FFT** 수행
6. **두 가지 이미지 저장**:
   - `Y/`: 1280x720 원본 이미지
   - `x/`: 256x144 저주파 이미지

```bash
python 01_YUV420_extractor.py
```

**출력 구조:**
```
work_dir/
├── video_scene_0001/
│   ├── Y/
│   │   ├── frame_00000001.png  # 1280x720
│   │   └── ...
│   └── x/
│       ├── frame_00000001.png  # 256x144
│       └── ...
├── video_scene_0002/
│   ├── Y/
│   └── x/
└── ...
```

### 2. GAN 모델 학습 (`03_train.py`)

**학습 데이터:**
- Input (x): 256x144 저주파 이미지
- Target (y): 1280x720 원본 이미지

**모델 구조:**
- **Generator**: 256x144 → 1280x720 (5배 업스케일)
  - Residual blocks
  - PixelShuffle upsampling
- **Discriminator**: 1280x720 이미지 진위 판별

```bash
python 03_train.py
```

### 3. 이미지 복원 (`04_recon.py`)

학습된 Generator로 저주파 이미지를 고해상도로 복원

```bash
python 04_recon.py
```

**출력:**
```
work_dir/
└── video_scene_0001/
    ├── Y/              # 원본
    ├── x/              # 저주파 입력
    └── y_recon/        # 복원된 고해상도
```

## ⚙️ 설정 (`config.py`)

모든 설정은 `config.py`에서 관리합니다.

```python
# GPU 사용 여부
USE_GPU = True

# Scene 검출 설정
class YUV420ExtractorConfig:
    BATCH_SIZE = 2400               # 한 번에 처리할 프레임 수
    MIN_SCENE_FRAMES = 240          # 최소 scene 길이
    SIMILARITY_THRESHOLD = 0.95     # 유사도 임계값 (낮을수록 민감)
    CROP_WIDTH = 256                # 저주파 크롭 너비
    CROP_HEIGHT = 144               # 저주파 크롭 높이

# 학습 설정
class TrainConfig:
    EPOCHS = 100
    BATCH_SIZE = 4
    LR_GENERATOR = 0.0002
    BASE_CHANNELS = 64
    NUM_RESIDUAL_BLOCKS = 8
```

## 📁 파일 구조

```
.
├── config.py                      # 통합 설정 파일
├── 01_YUV420_extractor.py        # Scene 검출 + 프레임 추출
├── 03_train.py                    # GAN 학습
├── 04_recon.py                    # 이미지 복원
├── videos/                        # 입력 비디오
├── work_dir/                      # 출력 (scene별 프레임)
└── models/                        # 학습된 모델
```

## 🔧 의존성

```bash
pip install torch torchvision numpy pillow ffmpeg-python tqdm
```

## 💡 주요 특징

### ✨ 주파수 공간 scene 검출
- 시간 영역 대신 **주파수 영역**에서 유사도 비교
- 색상 변화, 조명 변화에 더 강건
- FFT magnitude의 코사인 유사도 사용

### ⚡ 메모리 효율성
- 2400프레임 배치 처리
- Scene별 즉시 저장 및 메모리 해제
- GPU 메모리 최적화

### 🎯 통합 파이프라인
- **02_FFT_crop_IFFT.py 불필요**: 01에서 직접 x 이미지 생성
- 한 번의 실행으로 Y 이미지와 x 이미지 모두 생성

## 📊 성능 팁

### GPU 메모리 부족 시
```python
# config.py
YUV420ExtractorConfig.BATCH_SIZE = 1200  # 배치 크기 줄이기
```

### Scene이 너무 많이 분리될 때
```python
YUV420ExtractorConfig.SIMILARITY_THRESHOLD = 0.98  # 임계값 높이기
```

### Scene이 너무 적게 분리될 때
```python
YUV420ExtractorConfig.SIMILARITY_THRESHOLD = 0.90  # 임계값 낮추기
YUV420ExtractorConfig.MIN_SCENE_FRAMES = 120       # 최소 길이 줄이기
```

## 🚀 빠른 시작

1. 비디오 파일을 `videos/` 폴더에 넣기
2. 설정 확인: `python config.py`
3. Scene 검출 및 프레임 추출: `python 01_YUV420_extractor.py`
4. GAN 학습: `python 03_train.py`
5. 이미지 복원: `python 04_recon.py`

---

**Note:** 02_FFT_crop_IFFT.py는 더 이상 필요하지 않습니다. 01번 스크립트에서 Y와 x 이미지를 모두 생성합니다.
