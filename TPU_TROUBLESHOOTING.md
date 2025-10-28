# TPU 문제 해결 가이드

## 일반적인 문제와 해결 방법

### 1. nprocs 에러

**에러 메시지:**
```
ValueError: Unsupported nprocs (8). Please use nprocs=1 or None (default).
```

**원인:**
최신 torch_xla (PyTorch XLA 2.0+)에서는 `nprocs` 파라미터 사용 방식이 변경되었습니다.

**해결 방법:**
✅ **이미 수정됨** - `03_train_FastSRGAN.py`가 업데이트되어 `nprocs=None` 사용

**추가 설정 (선택사항):**
```bash
# TPU 디바이스 수를 제한하려면 환경 변수 사용
export TPU_NUM_DEVICES=4  # 4개 디바이스만 사용
export TPU_NUM_DEVICES=8  # 8개 디바이스 사용

# Python에서 실행 전 설정
import os
os.environ['TPU_NUM_DEVICES'] = '4'
```

---

### 2. torch_xla import 에러

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

### 3. TPU 디바이스를 찾을 수 없음

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
python -c "import torch_xla; import torch_xla.core.xla_model as xm; print(xm.xla_device())"
```

---

### 4. 메모리 부족 (OOM)

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

### 5. 느린 학습 속도

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

### 6. 모델 저장 실패

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

### 7. SyntaxWarning (무시 가능)

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
import torch_xla.core.xla_model as xm

# 현재 디바이스
print(f"Device: {xm.xla_device()}")

# 사용 가능한 디바이스 수
print(f"Number of devices: {xm.xrt_world_size()}")

# 현재 디바이스 순서
print(f"Ordinal: {xm.get_ordinal()}")

# Master 디바이스 확인
print(f"Is master: {xm.is_master_ordinal()}")
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
