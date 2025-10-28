# TPU v5 학습 가이드

## TPU v5 환경 설정

### 1. PyTorch XLA 설치
```bash
# TPU v5용 PyTorch XLA 설치
pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# 또는 최신 버전
pip install torch torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html
```

### 2. 설정 변경
`FastSRGANconfig.py`에서 TPU 사용 설정:

```python
config.device = 'tpu'  # 또는 'auto'로 설정하면 자동 감지
config.tpu_cores = 8   # TPU v5 코어 수
```

### 3. 학습 실행
```bash
# 학습 시작
python 03_train_FastSRGAN.py
```

TPU가 감지되면 자동으로 멀티프로세싱이 활성화되어 8개 코어에서 병렬 학습이 진행됩니다.

### 4. 추론 실행
```bash
# 이미지 복원
python 04_recon_FastSRGAN.py

# 벤치마크
python 04_recon_FastSRGAN.py  # 메인 함수에서 benchmark_fast_srgan() 호출
```

## TPU v5 특징

### 장점
- **높은 처리량**: GPU 대비 높은 배치 처리 성능
- **메모리**: 16GB HBM per core (v5e는 8개 코어 = 128GB)
- **가격 대비 성능**: 학습 비용 효율적

### 주의사항
1. **데이터 로딩**: TPU는 `num_workers=0` 사용 권장
2. **그래디언트 동기화**: `xm.mark_step()`으로 주기적 동기화 필요
3. **모델 저장**: master ordinal에서만 저장 (멀티프로세싱 시)
4. **배치 크기**: TPU는 큰 배치 크기에 최적화됨 (8, 16, 32 등)

## 성능 최적화 팁

### 배치 크기 증가
```python
config.batch_size = 8  # GPU: 1~2, TPU: 8~32
```

### TPU 코어 수 조정
```python
config.tpu_cores = 8  # v5e: 8 cores, v5p: 8 cores per chip
```

### 학습 속도 향상
- 더 큰 배치 크기 사용
- `learning_rate`를 배치 크기에 비례하여 증가
- `num_workers=0` 유지 (TPU는 데이터 로딩이 다름)

## GPU/CPU로 복귀
```python
config.device = 'cuda'  # GPU
config.device = 'cpu'   # CPU
config.device = 'auto'  # 자동 감지 (TPU > CUDA > CPU 순)
```

## 트러블슈팅

### torch_xla를 찾을 수 없음
```bash
pip install torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html
```

### TPU 인식 안됨
```python
# Python에서 확인
import torch_xla.core.xla_model as xm
device = xm.xla_device()
print(device)  # xla:0이 출력되어야 함
```

### 느린 학습 속도
- 배치 크기를 8 이상으로 증가
- `xm.mark_step()` 빈도 조정 (현재 10 iteration마다)
- 데이터셋 크기 확인 (너무 작으면 오버헤드가 큼)
