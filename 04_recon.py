"""
DeblurGAN-v2 추론 스크립트
주파수 도메인 DC 투영
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

from config import config
from models import Generator, create_frequency_mask


def reconstruct():
    """모든 Y_low 이미지를 복원"""
    device = torch.device(config.device)
    print(f"Device: {device}")
    
    # 모델 로드
    generator = Generator().to(device)
    
    if os.path.exists(config.model_path_gen):
        generator.load_state_dict(torch.load(config.model_path_gen, map_location=device, weights_only=True))
        print(f"✓ Model loaded: {config.model_path_gen}")
    else:
        print(f"✗ Model not found: {config.model_path_gen}")
        return
    
    generator.eval()
    
    # CUDA 최적화
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("✓ CUDA optimizations enabled")
    
    # 모든 비디오 폴더 처리
    video_dirs = [d for d in os.listdir(config.work_dir) 
                  if os.path.isdir(os.path.join(config.work_dir, d))]
    
    total_frames = 0
    total_time = 0.0
    
    for video_name in video_dirs:
        input_dir = os.path.join(config.work_dir, video_name, config.input_channel)
        if not os.path.exists(input_dir):
            continue
        
        # 출력 디렉토리
        output_dir = os.path.join(config.work_dir, video_name, config.output_channel)
        os.makedirs(output_dir, exist_ok=True)
        
        # PNG 파일들
        png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
        
        if len(png_files) == 0:
            continue
        
        print(f"\n{video_name}: {len(png_files)} frames")
        
        # 주파수 마스크 생성
        first_img = Image.open(png_files[0]).convert('L')
        H, W = first_img.size[1], first_img.size[0]
        freq_mask = create_frequency_mask((H, W), center_ratio=0.1, device=device)
        
        video_start = time.time()
        
        with torch.no_grad():
            for img_path in tqdm(png_files, desc=video_name):
                frame_start = time.time()
                
                # Load
                img = Image.open(img_path).convert('L')
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
                
                # Inference with DC projection
                output = generator.forward_with_dc(img_tensor, freq_mask)
                
                # Save
                output_array = output.squeeze().cpu().numpy()
                output_array = np.clip(output_array * 255.0, 0, 255).astype(np.uint8)
                output_img = Image.fromarray(output_array, mode='L')
                
                output_path = os.path.join(output_dir, os.path.basename(img_path))
                output_img.save(output_path, compress_level=0)
                
                frame_time = time.time() - frame_start
                total_time += frame_time
        
        video_time = time.time() - video_start
        fps = len(png_files) / video_time
        total_frames += len(png_files)
        
        print(f"✓ {fps:.2f} FPS ({video_time:.2f}s)")
    
    # 전체 통계
    if total_frames > 0:
        avg_fps = total_frames / total_time
        avg_time = total_time / total_frames
        
        print(f"\n{'='*60}")
        print(f"Total: {total_frames} frames, {total_time:.2f}s")
        print(f"Average: {avg_fps:.2f} FPS, {avg_time*1000:.2f}ms/frame")
        print(f"{'='*60}")


def benchmark(resolution=(576, 1024), num_frames=100):
    """속도 벤치마크"""
    device = torch.device(config.device)
    print(f"\n{'='*60}")
    print(f"Benchmark: {resolution[1]}x{resolution[0]}, {num_frames} frames")
    print(f"{'='*60}")
    
    # 모델 로드
    generator = Generator().to(device)
    
    if os.path.exists(config.model_path_gen):
        generator.load_state_dict(torch.load(config.model_path_gen, map_location=device, weights_only=True))
    else:
        print("⚠ Model not found, using random weights")
    
    generator.eval()
    
    # CUDA 최적화
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # 더미 입력
    dummy_input = torch.randn(1, 1, resolution[0], resolution[1]).to(device)
    freq_mask = create_frequency_mask(resolution, center_ratio=0.1, device=device)
    
    # 워밍업
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = generator.forward_with_dc(dummy_input, freq_mask)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 벤치마크
    print("Benchmarking...")
    start_time = time.time()
    
    with torch.no_grad():
        for _ in tqdm(range(num_frames)):
            output = generator.forward_with_dc(dummy_input, freq_mask)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    fps = num_frames / total_time
    ms = (total_time / num_frames) * 1000
    
    print(f"\n{'='*60}")
    print(f"FPS: {fps:.2f}")
    print(f"Time/frame: {ms:.2f}ms")
    
    if device.type == 'cuda':
        mem_alloc = torch.cuda.memory_allocated(device) / (1024**3)
        mem_reserved = torch.cuda.memory_reserved(device) / (1024**3)
        print(f"GPU Memory: {mem_alloc:.2f}GB / {mem_reserved:.2f}GB")
    
    # 다른 해상도 추정
    print(f"\nEstimated performance:")
    base_pixels = resolution[0] * resolution[1]
    
    for h, w in [(480, 854), (576, 1024), (720, 1280), (1080, 1920)]:
        pixels = h * w
        scale = pixels / base_pixels
        est_ms = ms * scale
        est_fps = 1000 / est_ms
        print(f"  {w}x{h}: ~{est_fps:.1f} FPS ({est_ms:.1f}ms)")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    print("="*60)
    print("DeblurGAN-v2 Reconstruction & Benchmark")
    print("="*60)
    
    # 복원
    reconstruct()
    
    # 벤치마크
    benchmark(resolution=(576, 1024), num_frames=100)
    
    print("\n✓ All tasks completed!")
