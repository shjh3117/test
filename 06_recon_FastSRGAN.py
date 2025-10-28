"""
Fast-SRGAN 모델을 사용한 Y_low 복원 및 벤치마크
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
    """Fast-SRGAN Generator 모델 로드"""
    device = torch.device(config.device)
    generator = FastSRGANGenerator().to(device)
    
    # T4 GPU 최적화 (추론용)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    model_path = config.model_path_gen
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Fast-SRGAN Generator loaded from {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return None, None
    
    generator.eval()
    return generator, device

def reconstruct_images_fast_srgan():
    """Fast-SRGAN을 사용한 이미지 복원"""
    generator, device = load_fast_srgan_model()
    if generator is None:
        return
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Using FP16: {config.use_amp}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
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
    resolutions = [(480, 854), (720, 1280), (1080, 1920)]
    base_pixels = resolution[0] * resolution[1]
    
    print(f"\n=== Estimated Performance at Different Resolutions ===")
    for h, w in resolutions:
        pixels = h * w
        scale_factor = pixels / base_pixels
        estimated_time = avg_time_per_frame * scale_factor
        estimated_fps = 1.0 / estimated_time if estimated_time > 0 else float('inf')
        
        print(f"{w}x{h}: ~{estimated_fps:.1f} FPS ({estimated_time*1000:.1f}ms per frame)")

def compare_with_espcn():
    """ESPCN과 Fast-SRGAN 성능 비교"""
    print(f"\n=== Model Comparison ===")
    
    # Fast-SRGAN 정보
    generator, device = load_fast_srgan_model()
    if generator:
        fast_srgan_params = sum(p.numel() for p in generator.parameters())
        print(f"Fast-SRGAN Parameters: {fast_srgan_params:,}")
    
    # 추정 비교 (실제 ESPCN 모델이 있다면 정확한 비교 가능)
    print(f"ESPCN Parameters: ~300K-1M (estimated)")
    print(f"Fast-SRGAN Parameters: ~{fast_srgan_params//1000}K")
    print(f"\nQuality: Fast-SRGAN > ESPCN (GAN-based)")
    print(f"Speed: ESPCN > Fast-SRGAN (simpler model)")
    print(f"Memory: Fast-SRGAN uses more memory")
    print(f"Training: Fast-SRGAN requires more training time")

if __name__ == "__main__":
    print("=== Fast-SRGAN Image Reconstruction ===")
    reconstruct_images_fast_srgan()
    
    print("\n=== Fast-SRGAN Speed Benchmark ===")
    benchmark_fast_srgan()
    
    print("\n=== Model Comparison ===")
    compare_with_espcn()