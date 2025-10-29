"""
Fast-SRGAN 모델을 사용한 Y_low 복원 및 성능 벤치마크

이 스크립트는 훈련된 Fast-SRGAN 모델을 사용하여:
1. Y_low 채널 이미지들을 고품질로 복원
2. 모델의 추론 속도 벤치마크 수행
3. 다른 모델과의 성능 비교 분석

주요 기능:
- 혼합 정밀도(FP16) 추론 지원으로 속도 향상
- 배치별 처리로 메모리 효율성 확보
- 실시간 FPS 모니터링
- 다양한 해상도에서의 성능 예측
"""

import os
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
from torch.amp import autocast

from FastSRGANconfig import fast_srgan_config as config
from FastSRGAN_models import FastSRGANGenerator

def load_fast_srgan_model():
    """
    Fast-SRGAN Generator 모델을 로드하고 추론을 위해 최적화
    
    Returns:
        tuple: (generator_model, device) 또는 (None, None) if 로드 실패
        
    최적화 내용:
    - CUDA 벤치마크 모드 활성화 (일관된 입력 크기에서 속도 향상)
    - TensorFloat-32 (TF32) 활성화 (Ampere GPU에서 성능 향상)
    - 모델을 평가 모드로 설정 (배치 정규화 및 드롭아웃 비활성화)
    """
    device = torch.device(config.device)
    generator = FastSRGANGenerator().to(device)
    
    # T4 GPU 최적화 (추론용)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # 고정 입력 크기에서 성능 향상
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    model_path = config.model_path_gen
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Fast-SRGAN Generator loaded from {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return None, None
    
    generator.eval()  # 평가 모드로 설정 (배치 정규화 등 비활성화)
    return generator, device

def reconstruct_images_fast_srgan():
    """Fast-SRGAN을 사용한 이미지 복원"""
    generator, device = load_fast_srgan_model()
    if generator is None:
        return
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Using FP16: {config.use_amp}")
    
    # 모든 비디오 폴더 찾기
    video_dirs = [d for d in os.listdir(config.work_dir) 
                  if os.path.isdir(os.path.join(config.work_dir, d))]
    
    total_processed = 0
    total_time = 0.0
    
    for video_name in video_dirs:
        input_dir = os.path.join(config.work_dir, video_name, config.input_channel)
        if not os.path.exists(input_dir):
            continue
        
        # 출력 디렉토리 생성
        output_dir = os.path.join(config.work_dir, video_name, config.output_channel)
        os.makedirs(output_dir, exist_ok=True)
        
        # 모든 PNG 파일 처리
        png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
        
        print(f"Processing {video_name}: {len(png_files)} frames")
        
        video_start_time = time.time()
        
        with torch.no_grad():
            for img_path in tqdm(png_files, desc=f"Fast-SRGAN {video_name}"):
                try:
                    frame_start_time = time.time()
                    
                    # 이미지 로드 및 전처리
                    img = Image.open(img_path).convert('L')
                    img_array = np.array(img, dtype=np.float32)
                    
                    # 정규화 (-1~1 범위)
                    if config.normalize:
                        img_array = (img_array / 255.0) * 2.0 - 1.0
                    else:
                        img_array = img_array / 255.0
                    
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
                    
                    # 모델 추론
                    if config.use_amp and device.type == 'cuda':
                        with autocast(device_type='cuda'):
                            output = generator(img_tensor)
                    else:
                        output = generator(img_tensor)
                    
                    # 후처리 및 저장
                    output_array = output.squeeze().cpu().numpy()
                    
                    # 정규화 해제
                    if config.normalize:
                        output_array = (output_array + 1.0) / 2.0 * 255.0
                    else:
                        output_array = output_array * 255.0
                    
                    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
                    output_img = Image.fromarray(output_array, mode='L')
                    
                    # 저장
                    output_path = os.path.join(output_dir, os.path.basename(img_path))
                    output_img.save(output_path, compress_level=0)
                    
                    # 시간 측정
                    frame_time = time.time() - frame_start_time
                    total_time += frame_time
                    total_processed += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        video_time = time.time() - video_start_time
        fps = len(png_files) / video_time if video_time > 0 else 0
        print(f"Completed: {video_name} ({fps:.2f} FPS)")
    
    # 전체 통계
    if total_processed > 0:
        avg_time_per_frame = total_time / total_processed
        avg_fps = 1.0 / avg_time_per_frame if avg_time_per_frame > 0 else 0
        
        print(f"\n=== Fast-SRGAN Processing Summary ===")
        print(f"Total frames processed: {total_processed}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average time per frame: {avg_time_per_frame*1000:.2f}ms")
        print(f"Average FPS: {avg_fps:.2f}")

def benchmark_fast_srgan():
    """Fast-SRGAN 속도 벤치마크"""
    generator, device = load_fast_srgan_model()
    if generator is None:
        return
    
    resolution = config.benchmark_resolution
    num_frames = config.benchmark_frames
    
    print(f"\n=== Fast-SRGAN Benchmark ===")
    print(f"Device: {device}")
    print(f"Using FP16: {config.use_amp and device.type == 'cuda'}")
    print(f"Resolution: {resolution[1]}x{resolution[0]}")
    print(f"Test frames: {num_frames}")
    
    # 더미 입력 생성 (-1~1 범위)
    dummy_input = torch.randn(1, 1, resolution[0], resolution[1]).to(device)
    if config.normalize:
        dummy_input = torch.tanh(dummy_input)  # -1~1 범위로 제한
    
    # 워밍업
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            if config.use_amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    _ = generator(dummy_input)
            else:
                _ = generator(dummy_input)
    
    # 메모리 정리
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 속도 측정
    print("Benchmarking...")
    start_time = time.time()
    
    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Benchmark"):
            if config.use_amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    output = generator(dummy_input)
            else:
                output = generator(dummy_input)
            
            # 중간 결과 확인 (처음과 마지막)
            if i == 0 or i == num_frames - 1:
                output_stats = {
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'mean': output.mean().item()
                }
                print(f"Frame {i+1} output stats: {output_stats}")
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    fps = num_frames / total_time
    avg_time_per_frame = total_time / num_frames
    
    # 메모리 사용량
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        memory_stats = f"Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved"
    else:
        memory_stats = "CPU mode"
    
    print(f"\n=== Benchmark Results ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {fps:.2f}")
    print(f"Time per frame: {avg_time_per_frame*1000:.2f}ms")
    print(f"{memory_stats}")
    
    # 다른 해상도에서의 예상 성능
    resolutions = [(480, 854), (576, 1024), (720, 1280)]
    base_pixels = resolution[0] * resolution[1]
    
    print(f"\n=== Estimated Performance at Different Resolutions ===")
    for h, w in resolutions:
        pixels = h * w
        scale_factor = pixels / base_pixels
        estimated_time = avg_time_per_frame * scale_factor
        estimated_fps = 1.0 / estimated_time if estimated_time > 0 else float('inf')
        
        print(f"{w}x{h}: ~{estimated_fps:.1f} FPS ({estimated_time*1000:.1f}ms per frame)")

if __name__ == "__main__":
    """
    메인 실행 함수
    
    실행 순서:
    1. 이미지 복원: 모든 Y_low 이미지를 Fast-SRGAN으로 복원
    2. 속도 벤치마크: 설정된 해상도에서 FPS 성능 측정
    3. 모델 비교: 다른 모델들과의 성능 비교 분석
    """
    
    print("=" * 60)
    print("Fast-SRGAN 이미지 복원 시작")
    print("=" * 60)
    reconstruct_images_fast_srgan()
    
    print("\n" + "=" * 60)
    print("Fast-SRGAN 속도 벤치마크 시작")
    print("=" * 60)
    benchmark_fast_srgan()
    
    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    print("=" * 60)