# TPU v5 지원 추가 - 변경 사항 요약

## 개요
Fast-SRGAN 프로젝트에 Google Cloud TPU v5 지원을 추가했습니다. 이제 GPU, CPU, TPU 환경 모두에서 학습 및 추론이 가능합니다.

## 변경된 파일

### 1. FastSRGANconfig.py
**변경 내용:**
- `device` 옵션에 'tpu' 추가
- `tpu_cores` 설정 추가 (기본값: 8)
- `__post_init__` 메서드에서 TPU 자동 감지 로직 추가

**주요 코드:**
```python
device: str = 'auto'  # 'auto', 'cuda', 'cpu', 'tpu'
tpu_cores: int = 8    # TPU 코어 수
```

### 2. 03_train_FastSRGAN.py
**변경 내용:**
- PyTorch XLA import 추가 (TPU 지원)
- 디바이스 초기화 로직 업데이트 (TPU 지원)
- 데이터 로더에 TPU용 MpDeviceLoader 래핑
- 옵티마이저 스텝에서 `xm.optimizer_step()` 사용
- 주기적 `xm.mark_step()` 호출 (그래디언트 동기화)
- 모델 저장 시 TPU master ordinal 체크
- TPU 멀티프로세싱 지원 (`xmp.spawn`)

**주요 기능:**
- TPU 8코어 멀티프로세싱 학습
- 자동 그래디언트 동기화
- TPU 최적화된 데이터 로딩

### 3. 04_recon_FastSRGAN.py
**변경 내용:**
- PyTorch XLA import 추가
- 모델 로딩 함수에 TPU 지원
- 추론 시 `xm.mark_step()` 추가
- 벤치마크 함수 TPU 지원

**주요 기능:**
- TPU에서 이미지 복원
- TPU 벤치마크 측정

## 새로 추가된 파일

### 1. requirements_tpu.txt
TPU 환경을 위한 패키지 요구사항
- torch_xla 설치 방법
- 기본 요구사항 목록

### 2. TPU_GUIDE.md
TPU v5 사용 가이드
- 환경 설정 방법
- 학습/추론 실행 방법
- 성능 최적화 팁
- 트러블슈팅

### 3. test_tpu_setup.py
TPU 환경 테스트 스크립트
- PyTorch XLA 설치 확인
- TPU 디바이스 접근 확인
- 기본 연산 테스트
- 모델 로딩 테스트

## 사용 방법

### TPU 환경 설정
```bash
# 1. PyTorch XLA 설치
pip install torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html

# 2. 환경 테스트
python test_tpu_setup.py

# 3. 설정 변경 (FastSRGANconfig.py)
device = 'tpu'  # 또는 'auto'
tpu_cores = 8

# 4. 학습 실행
python 03_train_FastSRGAN.py
```

### GPU/CPU 환경
기존과 동일하게 작동합니다. torch_xla가 없어도 자동으로 GPU/CPU로 폴백됩니다.

```python
device = 'cuda'  # GPU
device = 'cpu'   # CPU
device = 'auto'  # 자동 (TPU > CUDA > CPU)
```

## 호환성

### 지원 환경
- ✅ NVIDIA GPU (CUDA)
- ✅ CPU
- ✅ Google Cloud TPU v5e
- ✅ Google Cloud TPU v5p

### 주요 차이점

| 기능 | GPU | TPU |
|------|-----|-----|
| FP16/AMP | ✓ | - |
| 멀티프로세싱 | DataParallel | XLA MP |
| 배치 크기 권장 | 1-4 | 8-32 |
| num_workers | 4+ | 0 |
| 동기화 | automatic | xm.mark_step() |

## 성능 최적화

### TPU 최적화 설정
```python
# FastSRGANconfig.py
batch_size = 16           # TPU에서 큰 배치 크기 사용
learning_rate_gen = 1e-3  # 배치 크기에 비례하여 증가
tpu_cores = 8            # v5e 기준
```

### GPU 최적화 설정
```python
# FastSRGANconfig.py
batch_size = 2           # T4 GPU 기준
use_amp = True          # FP16 사용
num_workers = 4         # 데이터 로딩 병렬화
```

## 알려진 제한사항

1. **FP16/AMP**: TPU에서는 자동 mixed precision이 지원되지 않음 (FP32 학습)
2. **데이터 로딩**: TPU는 `num_workers=0` 권장
3. **디버깅**: TPU는 에러 메시지가 덜 직관적일 수 있음
4. **로컬 테스트**: TPU는 Google Cloud 환경에서만 사용 가능

## 테스트 상태

- ✅ 코드 구조 업데이트 완료
- ⚠️ TPU 실제 환경 테스트 필요 (Google Cloud TPU VM에서)
- ✅ GPU/CPU 하위 호환성 유지
- ✅ torch_xla 없어도 일반 동작 가능

## 다음 단계

1. Google Cloud TPU VM에서 실제 테스트
2. 성능 벤치마크 수집
3. 필요 시 배치 크기 및 학습률 튜닝
4. 메모리 사용량 최적화

## 문의사항
TPU 관련 문의는 `TPU_GUIDE.md` 참조 또는 Google Cloud TPU 문서를 확인하세요.
