"""
input_dir아래의 모든 영상들은 항상 yuv420영상이다.
Y프레임 간의 유사성(내적)을 계산하여 scene을 감지하고,
scene 단위로 Y, U, V를 추출하여 그레이스케일 무손실 압축(png)로 저장한다.
ex)
work_dir/input_scene_0001/Y/frame_00000001.png
work_dir/input_scene_0002/Y/frame_00000001.png
"""

import os
import glob
import ffmpeg
import torch
import numpy as np
from PIL import Image
import subprocess
import json

def get_video_info(video_path):
    """ffprobe를 사용하여 비디오 정보 추출"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info['streams'][0]
    width = int(stream['width'])
    height = int(stream['height'])
    nb_frames = int(stream.get('nb_frames', 0))
    return width, height, nb_frames

def extract_y_frames_to_tensor(video_path, device='cuda'):
    """
    비디오에서 Y 채널을 PNG로 추출 후 PyTorch Tensor로 로드 (고속 처리)
    """
    import tempfile
    import shutil
    from pathlib import Path
    
    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp(prefix='y_frames_')
    temp_path = Path(temp_dir)
    
    try:
        print(f"Extracting Y frames to temporary directory...")
        # ffmpeg로 Y 채널을 PNG로 추출 (무손실 압축)
        (
            ffmpeg
            .input(video_path)
            .filter('extractplanes', 'y')
            .output(str(temp_path / 'frame_%08d.png'), pix_fmt='gray', compression_level=0)
            .overwrite_output()
            .run(quiet=True)
        )
        
        # PNG 파일 목록 가져오기
        png_files = sorted(temp_path.glob('frame_*.png'))
        
        if not png_files:
            return None
        
        print(f"Loading {len(png_files)} frames to GPU...")
        
        # 첫 번째 이미지로 크기 확인
        first_img = Image.open(png_files[0])
        height, width = np.array(first_img).shape
        
        # 모든 프레임을 numpy array로 로드
        frames_list = []
        for png_file in png_files:
            img = Image.open(png_file)
            frame = np.array(img, dtype=np.uint8)
            frames_list.append(frame)
        
        # numpy array로 스택
        frames_np = np.stack(frames_list, axis=0)  # [N, H, W]
        
        # torch tensor로 변환 후 GPU로 전송
        frames_tensor = torch.from_numpy(frames_np).float().to(device)
        
        print(f"Loaded {frames_tensor.shape[0]} frames to GPU")
        
        return frames_tensor
        
    finally:
        # 임시 디렉토리 삭제
        print(f"Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)

def detect_scene_changes(y_frames, threshold=0.85):
    """
    Y 프레임 간의 정규화된 내적을 계산하여 scene 변화 감지
    
    Args:
        y_frames: [N, H, W] tensor
        threshold: 유사도 임계값 (이보다 낮으면 scene 변화로 감지)
    
    Returns:
        scene_boundaries: scene 시작 프레임 인덱스 리스트
    """
    n_frames = y_frames.shape[0]
    
    # 각 프레임을 벡터로 flatten하고 정규화
    frames_flat = y_frames.reshape(n_frames, -1)  # [N, H*W]
    frames_norm = torch.nn.functional.normalize(frames_flat, p=2, dim=1)
    
    # 연속된 프레임 간의 코사인 유사도 계산
    similarities = torch.sum(frames_norm[:-1] * frames_norm[1:], dim=1)  # [N-1]
    
    # 유사도가 임계값보다 낮은 지점을 scene 변화로 감지
    scene_changes = (similarities < threshold).cpu().numpy()
    
    # scene 시작 인덱스 (첫 프레임 + scene 변화 지점)
    scene_boundaries = [0]
    for i, is_change in enumerate(scene_changes):
        if is_change:
            scene_boundaries.append(i + 1)
    
    return scene_boundaries, similarities.cpu().numpy()

def extract_scene_yuv(video_path, scene_start, scene_end, output_dir):
    """
    특정 scene 구간의 Y, U, V 채널 추출
    """
    y_dir = os.path.join(output_dir, 'Y')
    u_dir = os.path.join(output_dir, 'U')
    v_dir = os.path.join(output_dir, 'V')
    
    os.makedirs(y_dir, exist_ok=True)
    os.makedirs(u_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)
    
    # Y 채널 추출
    (
        ffmpeg
        .input(video_path)
        .filter('select', f'gte(n,{scene_start})*lte(n,{scene_end-1})')
        .filter('setpts', 'N/FRAME_RATE/TB')
        .filter('extractplanes', 'y')
        .output(os.path.join(y_dir, 'frame_%08d.png'), 
                pix_fmt='gray', 
                compression_level=0,
                start_number=1,
                vsync='0')
        .overwrite_output()
        .run(quiet=True)
    )
    
    # U 채널 추출
    (
        ffmpeg
        .input(video_path)
        .filter('select', f'gte(n,{scene_start})*lte(n,{scene_end-1})')
        .filter('setpts', 'N/FRAME_RATE/TB')
        .filter('extractplanes', 'u')
        .output(os.path.join(u_dir, 'frame_%08d.png'), 
                pix_fmt='gray', 
                compression_level=0,
                start_number=1,
                vsync='0')
        .overwrite_output()
        .run(quiet=True)
    )
    
    # V 채널 추출
    (
        ffmpeg
        .input(video_path)
        .filter('select', f'gte(n,{scene_start})*lte(n,{scene_end-1})')
        .filter('setpts', 'N/FRAME_RATE/TB')
        .filter('extractplanes', 'v')
        .output(os.path.join(v_dir, 'frame_%08d.png'), 
                pix_fmt='gray', 
                compression_level=0,
                start_number=1,
                vsync='0')
        .overwrite_output()
        .run(quiet=True)
    )

def YUV420_extractor(input_dir='videos', output_dir='work_dir', similarity_threshold=0.85):
    """
    YUV420 비디오에서 Y 프레임 유사성 기반으로 scene을 감지하고,
    scene별로 Y, U, V 채널을 추출하여 PNG로 저장
    
    Args:
        input_dir: 입력 비디오 디렉토리
        output_dir: 출력 디렉토리
        similarity_threshold: scene 감지 임계값 (낮을수록 민감)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 입력 디렉토리에서 모든 비디오 파일 찾기
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n{'='*60}")
        print(f"Processing video: {video_name}")
        print(f"{'='*60}")
        
        try:
            # Y 프레임 추출 (GPU)
            print("Extracting Y frames to GPU memory...")
            y_frames = extract_y_frames_to_tensor(video_path, device=device)
            
            if y_frames is None:
                print(f"Failed to extract frames from {video_path}")
                continue
            
            print(f"Total frames: {y_frames.shape[0]}")
            
            # Scene 감지 (GPU accelerated)
            print("Detecting scene changes...")
            scene_boundaries, similarities = detect_scene_changes(y_frames, threshold=similarity_threshold)
            scene_boundaries.append(y_frames.shape[0])  # 마지막 프레임
            
            print(f"Detected {len(scene_boundaries)-1} scenes")
            print(f"Scene boundaries: {scene_boundaries}")
            
            # Scene별로 Y, U, V 추출
            for scene_idx in range(len(scene_boundaries) - 1):
                scene_start = scene_boundaries[scene_idx]
                scene_end = scene_boundaries[scene_idx + 1]
                scene_length = scene_end - scene_start
                
                scene_name = f"{video_name}_scene_{scene_idx+1:04d}"
                output_scene_dir = os.path.join(output_dir, scene_name)
                
                print(f"\nExtracting scene {scene_idx+1}/{len(scene_boundaries)-1}: "
                      f"frames {scene_start}-{scene_end-1} ({scene_length} frames)")
                
                extract_scene_yuv(video_path, scene_start, scene_end, output_scene_dir)
                
                print(f"  ✓ Saved to {output_scene_dir}")
            
            print(f"\n✓ Successfully processed {video_name}")
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # GPU 메모리 해제
            if 'y_frames' in locals():
                del y_frames
            torch.cuda.empty_cache()

if __name__ == "__main__":
    YUV420_extractor(similarity_threshold=0.9)
