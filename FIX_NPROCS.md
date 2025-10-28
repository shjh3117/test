# ✅ nprocs 에러 수정 완료

## 문제
```
ValueError: Unsupported nprocs (8). Please use nprocs=1 or None (default).
```

## 해결
`03_train_FastSRGAN.py`의 `nprocs=8` → `nprocs=None`으로 변경

## 변경 내역

### Before (이전)
```python
xmp.spawn(_mp_fn, args=(), nprocs=config.tpu_cores)  # ❌ 8 고정
```

### After (수정 후)
```python
xmp.spawn(_mp_fn, args=(), nprocs=None)  # ✅ 자동으로 모든 디바이스 사용
```

## 사용 방법

### 기본 사용 (모든 TPU 디바이스)
```bash
python 03_train_FastSRGAN.py
```
→ 자동으로 모든 사용 가능한 TPU 디바이스 활용

### 디바이스 수 제한
```bash
# 4개 디바이스만 사용
export TPU_NUM_DEVICES=4
python 03_train_FastSRGAN.py

# 또는 한 줄로
TPU_NUM_DEVICES=4 python 03_train_FastSRGAN.py
```

### Python 코드에서 설정
```python
import os
os.environ['TPU_NUM_DEVICES'] = '4'

# 그 다음 학습 실행
```

## 설정 변경사항

### FastSRGANconfig.py
- ❌ 제거: `tpu_cores: int = 8`
- ✅ 추가: 주석으로 환경 변수 사용법 안내

## 지금 바로 실행 가능

이제 다음 명령으로 바로 학습을 시작할 수 있습니다:

```bash
cd /content/test
python 03_train_FastSRGAN.py
```

TPU가 자동으로 감지되고 모든 사용 가능한 디바이스를 활용합니다!

## 추가 문서
- `TPU_TROUBLESHOOTING.md` - 상세 문제 해결 가이드
- `TPU_GUIDE.md` - TPU 사용 가이드
- `README_TPU.md` - 빠른 시작 가이드
