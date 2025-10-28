# TPU 문제 해결 가이드

## 일반적인 문제와 해결 방법

### 1. TPU 초기화 에러 (멀티프로세싱 관련)

**에러 메시지:**
```
RuntimeError: Bad StatusOr access: UNKNOWN: TPU initialization failed
Check failed: !g_computation_client_initialized
```

**원인:**
멀티프로세싱 spawn이 TPU 단일 호스트 환경에서 충돌을 일으킴

**해결 방법:**
✅ **이미 수정됨** - `xmp.spawn` 제거, 단일 프로세스 모드로 변경
- TPU는 단일 프로세스에서 자동으로 모든 코어를 활용합니다
- 멀티프로세싱 없이도 모든 TPU 디바이스가 활성화됩니다

---

### 2. nprocs 에러 (구버전)

### 2. nprocs 에러 (구버전)

**에러 메시지:**
```
ValueError: Unsupported nprocs (8). Please use nprocs=1 or None (default).
```

**원인:**
이전 버전에서 사용하던 멀티프로세싱 방식

**해결 방법:**
✅ **이미 수정됨** - 멀티프로세싱 제거, 단일 프로세스 모드 사용

---

### 3. torch_xla import 에러

**에러 메시지:**
```
ModuleNotFoundError: No module named 'torch_xla'
```

**해결 방법:**
```bash
# PyTorch XLA 설치
pip install torch torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html

# 또는 특정 버전
pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

---

### 4. TPU 디바이스를 찾을 수 없음

**에러 메시지:**
```
RuntimeError: No TPU devices found
```

**확인 사항:**
1. Google Cloud TPU VM에서 실행 중인가?
2. TPU가 올바르게 할당되었는가?

**해결 방법:**
```bash
# TPU 상태 확인
gcloud compute tpus list

# TPU 디바이스 확인
python -c "import torch_xla; print(torch_xla.device())"
```

---

### 5. 메모리 부족 (OOM)

**에러 메시지:**
```
RuntimeError: XLA out of memory
```

**해결 방법:**
```python
# FastSRGANconfig.py에서 배치 크기 줄이기
batch_size = 4  # 8에서 4로 감소

# 또는 이미지 크기 확인
# 너무 큰 이미지는 리사이즈
```

---

### 6. 느린 학습 속도

**증상:**
TPU를 사용하는데 GPU보다 느림

**원인:**
1. 배치 크기가 너무 작음
2. 데이터 로딩 병목
3. 빈번한 CPU-TPU 데이터 전송

**해결 방법:**
```python
# 1. 배치 크기 증가 (TPU는 큰 배치에 최적화)
batch_size = 16  # 또는 32

# 2. num_workers 확인
num_workers = 0  # TPU는 0 권장

# 3. mark_step 빈도 조정
# 03_train_FastSRGAN.py에서:
if use_tpu and batch_idx % 20 == 0:  # 10에서 20으로
    xm.mark_step()
```

---

### 7. 모델 저장 실패

**에러 메시지:**
```
Permission denied or File not found
```

**해결 방법:**
```python
# 저장 경로 확인
import os
print(os.getcwd())  # 현재 디렉토리 확인

# 권한이 있는 디렉토리에 저장
config.model_path_gen = '/tmp/fast_srgan_generator_best.pth'
```

---

### 8. SyntaxWarning (무시 가능)

**경고 메시지:**
```
SyntaxWarning: invalid escape sequence '\_'
```

**설명:**
torch_xla 라이브러리 내부의 경고로, 학습에는 영향 없음

**무시하려면:**
```bash
python -W ignore::SyntaxWarning 03_train_FastSRGAN.py
```

---

## 디버깅 팁

### TPU 상태 확인
```python
import torch_xla
import torch_xla.core.xla_model as xm

# 현재 디바이스 (최신 API)
device = torch_xla.device()
print(f"Device: {device}")

# 디바이스 개수
import torch_xla.runtime as xr
print(f"Number of devices: {xr.world_size()}")

# 현재 디바이스 순서
print(f"Global ordinal: {xr.global_ordinal()}")
```
```

### 메모리 사용량 확인
```python
# TPU 메모리는 torch.cuda와 다르게 동작
# 대신 학습 중 로그 확인
```

### 성능 프로파일링
```bash
# TPU 프로파일링 활성화
export TPU_METRICS_DEBUG=1

# 상세 로깅
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
```

---

## 추가 리소스

- [PyTorch XLA 공식 문서](https://pytorch.org/xla/)
- [Google Cloud TPU 문서](https://cloud.google.com/tpu/docs)
- [TPU 성능 가이드](https://cloud.google.com/tpu/docs/performance-guide)

---

## 문제가 계속되면

1. `test_tpu_setup.py` 실행하여 환경 확인
2. torch_xla 버전 확인: `pip show torch_xla`
3. Google Cloud TPU 지원팀 문의
