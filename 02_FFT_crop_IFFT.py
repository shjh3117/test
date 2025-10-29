"""
1. input_dir아래의 각 영상들의 특정 채널을 선택한다.
2. fft및 shift를 한다
3. Y채널 이미지를 특정 해상도로(256x144) 중앙 패치를 자른다.
4. 원본 Y채널 이미지의 해상도로 제로채딩을 시횅한다.
5. 다시 역shift 및 역fft를 시행한다.
6. 다음 예시처럼 그레이스케일 무손실 압축(png)로 저장한다.
ex)
work_dir/input/Y_low/frame_00000001.png
work_dir/input2/Y_low/frame_00000001.png
"""

import os
import glob
import torch
import torch.fft
import numpy as np
from PIL import Image

def FFT_crop_IFFT(input_dir='work_dir', output_dir='work_dir', channel='Y'):
    """
    FFT를 사용하여 주파수 도메인에서 크롭하고 IFFT로 복원하는 함수
    """
    # CUDA 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 입력 디렉토리에서 모든 비디오 폴더 찾기
    video_dirs = [d for d in os.listdir(input_dir) 
                  if os.path.isdir(os.path.join(input_dir, d))]
    
    if not video_dirs:
        print(f"No video directories found in {input_dir}")
        return
    
    for video_name in video_dirs:
        video_path = os.path.join(input_dir, video_name)
        channel_path = os.path.join(video_path, channel)
        
        if not os.path.exists(channel_path):
            print(f"Channel {channel} not found in {video_path}")
            continue
            
        print(f"Processing video: {video_name}, channel: {channel}")
        
        # 출력 디렉토리 생성
        output_video_dir = os.path.join(output_dir, video_name)
        output_channel_dir = os.path.join(output_video_dir, f"{channel}_low")
        os.makedirs(output_channel_dir, exist_ok=True)
        
        # 채널 디렉토리의 모든 PNG 파일 찾기
        frame_files = sorted(glob.glob(os.path.join(channel_path, "*.png")))
        
        if not frame_files:
            print(f"No PNG files found in {channel_path}")
            continue
        
        print(f"Found {len(frame_files)} frames to process")
        
        for i, frame_path in enumerate(frame_files):
            try:
                # 1. 이미지 로드
                image = Image.open(frame_path).convert('L')
                image_array = np.array(image, dtype=np.float32)
                
                # 원본 해상도 저장
                original_height, original_width = image_array.shape
                
                # numpy 배열을 torch 텐서로 변환하고 GPU로 이동
                image_tensor = torch.from_numpy(image_array).to(device)
                
                # 2. FFT 및 shift 수행
                fft_image = torch.fft.fft2(image_tensor)
                fft_shifted = torch.fft.fftshift(fft_image)
                
                # 3. 중앙 패치 크롭 (256x144)
                crop_height, crop_width = 144, 256
                
                center_y, center_x = original_height // 2, original_width // 2
                start_y = center_y - crop_height // 2
                end_y = start_y + crop_height
                start_x = center_x - crop_width // 2
                end_x = start_x + crop_width
                
                # 크롭된 주파수 스펙트럼
                cropped_fft = fft_shifted[start_y:end_y, start_x:end_x]
                
                # 4. 원본 해상도로 제로 패딩
                # 새로운 텐서를 생성하고 중앙에 크롭된 데이터 배치
                padded_fft = torch.zeros((original_height, original_width), 
                                       dtype=torch.complex64, device=device)
                
                padded_start_y = center_y - crop_height // 2
                padded_end_y = padded_start_y + crop_height
                padded_start_x = center_x - crop_width // 2
                padded_end_x = padded_start_x + crop_width
                
                padded_fft[padded_start_y:padded_end_y, 
                          padded_start_x:padded_end_x] = cropped_fft
                
                # 5. 역 shift 및 역 FFT 수행
                fft_ishifted = torch.fft.ifftshift(padded_fft)
                reconstructed = torch.fft.ifft2(fft_ishifted)
                
                # 실수부만 취하고 값 범위를 [0, 255]로 클램핑
                reconstructed_real = torch.real(reconstructed)
                reconstructed_real = torch.clamp(reconstructed_real, 0, 255)
                
                # CPU로 이동하고 numpy 배열로 변환
                result_array = reconstructed_real.cpu().numpy().astype(np.uint8)
                
                # 6. PNG로 저장
                result_image = Image.fromarray(result_array, mode='L')
                frame_name = os.path.basename(frame_path)
                output_path = os.path.join(output_channel_dir, frame_name)
                result_image.save(output_path, compress_level=0)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(frame_files)} frames")
                    
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
                continue
        
        print(f"Completed processing {video_name}, channel: {channel}")

if __name__ == "__main__":
    FFT_crop_IFFT()