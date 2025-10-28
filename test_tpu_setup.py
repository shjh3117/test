"""
TPU 환경 테스트 스크립트
TPU가 정상적으로 설정되었는지 확인
"""

import sys

def test_torch():
    """PyTorch 설치 확인"""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def test_torch_xla():
    """PyTorch XLA (TPU 지원) 설치 확인"""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        print(f"✓ PyTorch XLA installed")
        return True, xm
    except ImportError:
        print("✗ PyTorch XLA not installed")
        print("  Install with: pip install torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html")
        return False, None

def test_tpu_device(xm):
    """TPU 디바이스 접근 확인"""
    try:
        import torch_xla
        device = torch_xla.device()
        print(f"✓ TPU device available: {device}")
        return True, device
    except Exception as e:
        print(f"✗ TPU device not available: {e}")
        return False, None

def test_basic_operation(xm, device):
    """기본 TPU 연산 테스트"""
    try:
        import torch
        # 간단한 텐서 연산
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        z = x + y
        xm.mark_step()  # TPU 동기화
        
        # CPU로 가져와서 확인
        result = z.cpu()
        print(f"✓ Basic TPU operations work")
        print(f"  Sample result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"✗ TPU operations failed: {e}")
        return False

def test_model_loading(device):
    """모델 로딩 테스트"""
    try:
        from FastSRGAN_models import FastSRGANGenerator
        model = FastSRGANGenerator().to(device)
        print(f"✓ FastSRGAN model loaded on TPU")
        
        import torch
        # 간단한 forward pass
        dummy_input = torch.randn(1, 1, 256, 256).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def main():
    print("="*60)
    print("TPU v5 환경 테스트")
    print("="*60)
    
    # 1. PyTorch 확인
    if not test_torch():
        print("\n❌ PyTorch가 설치되지 않았습니다.")
        return
    
    # 2. PyTorch XLA 확인
    xla_ok, xm = test_torch_xla()
    if not xla_ok:
        print("\n❌ PyTorch XLA가 설치되지 않았습니다.")
        print("GPU/CPU 환경에서는 TPU 기능을 사용할 수 없지만 일반 학습은 가능합니다.")
        return
    
    # 3. TPU 디바이스 확인
    device_ok, device = test_tpu_device(xm)
    if not device_ok:
        print("\n❌ TPU 디바이스를 찾을 수 없습니다.")
        print("Google Cloud TPU VM에서 실행하고 있는지 확인하세요.")
        return
    
    # 4. 기본 연산 테스트
    if not test_basic_operation(xm, device):
        print("\n❌ TPU 연산에 실패했습니다.")
        return
    
    # 5. 모델 로딩 테스트
    if not test_model_loading(device):
        print("\n⚠ 모델 로딩에 실패했습니다. FastSRGAN 모델 파일을 확인하세요.")
    
    print("\n" + "="*60)
    print("✅ TPU 환경이 정상적으로 설정되었습니다!")
    print("="*60)
    print("\n다음 명령으로 학습을 시작할 수 있습니다:")
    print("  python 03_train_FastSRGAN.py")
    print("\n설정 확인:")
    print("  - FastSRGANconfig.py에서 device='tpu' 또는 'auto' 설정")
    print("  - 배치 크기를 8 이상으로 증가 권장")

if __name__ == "__main__":
    main()
