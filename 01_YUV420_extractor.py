"""
input_dir아래의 모든 영상들은 항상 yuv420영상이다.
Y프레임 간의 유사성(내적)을 계산하여 scene을 감지하고,
scene 단위로 Y, U, V를 추출하여 그레이스케일 무손실 압축(png)로 저장한다.
메모리 최적화: 청크 단위 처리, 즉시 메모리 해제, 배치 처리
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
import gc
from pathlib import Path

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

def extract_y_frames_to_tensor(video_path, device='cuda', chunk_size=100):
    """
    비디오에서 Y 채널을 청크 단위로 추출하여 메모리 효율적 처리
    
    Args:
        video_path: 비디오 파일 경로
        device: 'cuda' 또는 'cpu'
        chunk_size: 한 번에 처리할 프레임 수 (메모리 사용량 조절)
    
    Returns:
        y_frames: [N, H, W] tensor (GPU/CPU)
    """
    import tempfile
    import shutil
    
    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp(prefix='y_frames_')
    temp_path = Path(temp_dir)
    
    try:
        print(f"Extracting Y frames to temporary directory...")
        # ffmpeg로 Y 채널을 PNG로 추출 (무손실 압축)
        process = (
            ffmpeg
            .input(video_path)
            .filter('extractplanes', 'y')
            .output(str(temp_path / 'frame_%08d.png'), pix_fmt='gray', compression_level=0)
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        
        # 프로세스 완료 대기
        stdout, stderr = process.communicate()
        
        # PNG 파일 목록 가져오기
        png_files = sorted(temp_path.glob('frame_*.png'))
        
        if not png_files:
            return None
        
        total_frames = len(png_files)
        print(f"Found {total_frames} frames, loading in chunks of {chunk_size}...")
        
        # 첫 번째 이미지로 크기 확인
        with Image.open(png_files[0]) as first_img:
            height, width = np.array(first_img).shape
        
        # 청크 단위로 프레임 로드
        frame_chunks = []
        
        for i in range(0, total_frames, chunk_size):
            end_idx = min(i + chunk_size, total_frames)
            chunk_files = png_files[i:end_idx]
            
            # 청크 내 프레임 로드
            chunk_frames = []
            for png_file in chunk_files:
                with Image.open(png_file) as img:
                    frame = np.array(img, dtype=np.uint8)
                    chunk_frames.append(frame)
                # 즉시 메모리에서 제거
                del img
            
            # numpy array로 변환
            chunk_np = np.stack(chunk_frames, axis=0)
            del chunk_frames  # 리스트 메모리 해제
            
            # torch tensor로 변환 후 디바이스로 전송
            chunk_tensor = torch.from_numpy(chunk_np).float().to(device)
            del chunk_np  # numpy 메모리 해제
            
            frame_chunks.append(chunk_tensor)
            
            print(f"  Loaded chunk {i//chunk_size + 1}/{(total_frames + chunk_size - 1)//chunk_size}")
            
            # 메모리 정리
            gc.collect()
        
        # 모든 청크를 하나의 텐서로 결합
        frames_tensor = torch.cat(frame_chunks, dim=0)
        del frame_chunks  # 청크 리스트 메모리 해제
        
        # 메모리 정리
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"Loaded {frames_tensor.shape[0]} frames to {device}")
        
        return frames_tensor
        
    finally:
        # 임시 디렉토리 삭제
        print(f"Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        gc.collect()

def detect_scene_changes(y_frames, threshold=0.85, batch_size=50):
    """
    Y 프레임 간의 정규화된 내적을 배치 단위로 계산하여 scene 변화 감지 (메모리 최적화)
    
    Args:
        y_frames: [N, H, W] tensor
        threshold: 유사도 임계값 (이보다 낮으면 scene 변화로 감지)
        batch_size: 한 번에 처리할 프레임 쌍 수
    
    Returns:
        scene_boundaries: scene 시작 프레임 인덱스 리스트
    """
    n_frames = y_frames.shape[0]
    
    if n_frames <= 1:
        return [0], []
    
    # 각 프레임을 벡터로 flatten하고 정규화
    frames_flat = y_frames.reshape(n_frames, -1)  # [N, H*W]
    frames_norm = torch.nn.functional.normalize(frames_flat, p=2, dim=1)
    
    # 배치 단위로 연속된 프레임 간의 코사인 유사도 계산
    similarities_list = []
    
    for i in range(0, n_frames - 1, batch_size):
        end_idx = min(i + batch_size, n_frames - 1)
        
        # 현재 프레임들과 다음 프레임들 간의 유사도 계산
        curr_batch = frames_norm[i:end_idx]
        next_batch = frames_norm[i+1:end_idx+1]
        
        batch_similarities = torch.sum(curr_batch * next_batch, dim=1)
        similarities_list.append(batch_similarities.cpu())
        
        # 배치 처리 후 메모리 정리
        del curr_batch, next_batch, batch_similarities
        if y_frames.is_cuda:
            torch.cuda.empty_cache()
    
    # 모든 유사도 결합
    similarities = torch.cat(similarities_list).numpy()
    del similarities_list
    gc.collect()
    
    # 유사도가 임계값보다 낮은 지점을 scene 변화로 감지
    scene_changes = (similarities < threshold)
    
    # scene 시작 인덱스 (첫 프레임 + scene 변화 지점)
    scene_boundaries = [0]
    for i, is_change in enumerate(scene_changes):
        if is_change:
            scene_boundaries.append(i + 1)
    
    return scene_boundaries, similarities

def extract_scene_yuv(video_path, scene_start, scene_end, output_dir):
    """
    특정 scene 구간의 Y, U, V 채널 추출 (프로세스 안전 처리)
    """
    y_dir = os.path.join(output_dir, 'Y')
    u_dir = os.path.join(output_dir, 'U')
    v_dir = os.path.join(output_dir, 'V')
    
    os.makedirs(y_dir, exist_ok=True)
    os.makedirs(u_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)
    
    # Y 채널 추출
    try:
        process = (
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
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        process.communicate()
        process.wait()
    finally:
        if 'process' in locals() and process.poll() is None:
            process.kill()
    
    # U 채널 추출
    try:
        process = (
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
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        process.communicate()
        process.wait()
    finally:
        if 'process' in locals() and process.poll() is None:
            process.kill()
    
    # V 채널 추출
    try:
        process = (
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
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        process.communicate()
        process.wait()
    finally:
        if 'process' in locals() and process.poll() is None:
            process.kill()
        gc.collect()

def YUV420_extractor(input_dir='videos', output_dir='work_dir', similarity_threshold=0.85, 
                     chunk_size=100, batch_size=50):
    """
    YUV420 비디오에서 Y 프레임 유사성 기반으로 scene을 감지하고,
    scene별로 Y, U, V 채널을 추출하여 PNG로 저장
    메모리 최적화: 청크 단위 처리, 배치 연산, 즉시 해제
    
    Args:
        input_dir: 입력 비디오 디렉토리
        output_dir: 출력 디렉토리
        similarity_threshold: scene 감지 임계값 (낮을수록 민감)
        chunk_size: 프레임 로딩 청크 크기 (메모리 사용량 조절)
        batch_size: scene 감지 배치 크기 (메모리 사용량 조절)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Memory optimization: chunk_size={chunk_size}, batch_size={batch_size}")
    
    # 입력 디렉토리에서 모든 비디오 파일 찾기
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    for video_idx, video_path in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n{'='*60}")
        print(f"Processing video {video_idx+1}/{len(video_files)}: {video_name}")
        print(f"{'='*60}")
        
        y_frames = None
        
        try:
            # Y 프레임 추출 (청크 단위)
            print("Extracting Y frames with memory optimization...")
            y_frames = extract_y_frames_to_tensor(video_path, device=device, chunk_size=chunk_size)
            
            if y_frames is None:
                print(f"Failed to extract frames from {video_path}")
                continue
            
            print(f"Total frames: {y_frames.shape[0]}")
            
            # Scene 감지 (배치 단위)
            print("Detecting scene changes with memory optimization...")
            scene_boundaries, similarities = detect_scene_changes(
                y_frames, threshold=similarity_threshold, batch_size=batch_size
            )
            scene_boundaries.append(y_frames.shape[0])  # 마지막 프레임
            
            print(f"Detected {len(scene_boundaries)-1} scenes")
            print(f"Scene boundaries: {scene_boundaries}")
            
            # Y 프레임 메모리 해제 (scene 감지 완료 후)
            del y_frames
            y_frames = None
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            
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
                
                # Scene 추출 후 메모리 정리
                gc.collect()
            
            print(f"\n✓ Successfully processed {video_name}")
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # GPU/메모리 완전 해제
            if y_frames is not None:
                del y_frames
            if 'similarities' in locals():
                del similarities
            if 'scene_boundaries' in locals():
                del scene_boundaries
            
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            print(f"Memory cleanup completed for {video_name}")

if __name__ == "__main__":
    # 메모리 사용량에 따라 chunk_size와 batch_size 조정
    # GPU 메모리가 작으면 chunk_size=50, batch_size=25 등으로 줄이기
    YUV420_extractor(similarity_threshold=0.9, chunk_size=100, batch_size=50)
