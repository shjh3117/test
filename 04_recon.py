"""
ESPCN 모델을 사용한 Y_low 복원 (설정 파일 기반)
"""

import os
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast
from ESPCNconfig import config

class ESPCN(nn.Module):
    def __init__(self):
        super(ESPCN, self).__init__()
        
        # 설정에서 파라미터 가져오기
        num_features = config.num_features
        num_layers = config.num_layers
        kernel_size = config.kernel_size
        use_bn = config.use_batch_norm
        dropout_rate = config.dropout_rate
        
        # Sequential 모델로 간소화
        layers = []
        
        # 첫 번째 레이어
        layers.append(nn.Conv2d(1, num_features, kernel_size=kernel_size, padding=kernel_size//2))
        
        if config.activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        
        # 중간 레이어들
        for i in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            
            if config.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))
            
            if use_bn:
                layers.append(nn.BatchNorm2d(num_features))
            
            if dropout_rate > 0.0 and i % 2 == 1:
                layers.append(nn.Dropout2d(dropout_rate))
        
        # 마지막 레이어
        layers.append(nn.Conv2d(num_features, 1, kernel_size=3, padding=1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def load_model():
    device = torch.device(config.device)
    model = ESPCN().to(device)
    
    # T4 GPU 최적화 (추론용)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    model_path = config.model_path
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return None, None
    
    model.eval()
    return model, device

def reconstruct_images():
    model, device = load_model()
    if model is None:
        return
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Using FP16: {config.use_amp}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 모든 비디오 폴더 찾기
    video_dirs = [d for d in os.listdir(config.work_dir) 
                  if os.path.isdir(os.path.join(config.work_dir, d))]
    
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
        
        with torch.no_grad():
            for img_path in tqdm(png_files, desc=f"Reconstructing {video_name}"):
                try:
                    # 이미지 로드 및 전처리
                    img = Image.open(img_path).convert('L')
                    img_array = np.array(img, dtype=np.float32)
                    
                    if config.normalize:
                        img_array = img_array / 255.0
                    
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
                    
                    # 모델 추론
                    if config.use_amp and device.type == 'cuda':
                        with autocast(device_type='cuda'):
                            output = model(img_tensor)
                    else:
                        output = model(img_tensor)
                    
                    # 후처리 및 저장
                    output_array = output.squeeze().cpu().numpy()
                    
                    if config.normalize:
                        output_array = output_array * 255.0
                    
                    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
                    output_img = Image.fromarray(output_array, mode='L')
                    
                    # 저장
                    output_path = os.path.join(output_dir, os.path.basename(img_path))
                    output_img.save(output_path, compress_level=0)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        print(f"Completed: {video_name}")

def benchmark_speed():
    import time
    
    model, device = load_model()
    if model is None:
        return
    
    resolution = config.benchmark_resolution
    num_frames = config.benchmark_frames
    
    # 더미 입력 생성
    dummy_input = torch.randn(1, 1, resolution[0], resolution[1]).to(device)
    
    # 워밍업
    with torch.no_grad():
        for _ in range(10):
            if config.use_amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
    
    # 속도 측정
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_frames):
            if config.use_amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    fps = num_frames / total_time
    
    print(f"Benchmark Results:")
    print(f"Device: {device}")
    print(f"Using FP16: {config.use_amp and device.type == 'cuda'}")
    print(f"Resolution: {resolution[1]}x{resolution[0]}")
    print(f"Processed {num_frames} frames in {total_time:.2f}s")
    print(f"Average FPS: {fps:.2f}")
    print(f"Time per frame: {total_time/num_frames*1000:.2f}ms")

if __name__ == "__main__":
    # 이미지 복원
    reconstruct_images()
    
    # 속도 벤치마크
    print("\nSpeed Benchmark:")
    benchmark_speed()