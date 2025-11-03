"""
주파수 공간 기반 Scene 검출 및 Y 프레임 추출
1. 2400개의 Y 프레임을 GPU에 로드
2. FFT 및 shift 수행
3. 주파수 공간에서 유사도로 Scene 검출 (240프레임 동안 새 scene 없으면 그대로 scene)
4. Scene의 중앙 256x144 패치를 크롭하여 저주파 대역 확보
5. 역shift 및 역FFT 수행
6. 원본 Y 이미지(1280x720)와 저주파 이미지(256x144) 모두 PNG로 저장

출력:
work_dir/video_scene_0001/Y/frame_00000001.png (1280x720 원본)
work_dir/video_scene_0001/x/frame_00000001.png (256x144 저주파)
"""

import os
import glob
import ffmpeg
import torch
import torch.fft
import numpy as np
from PIL import Image
import subprocess
import json
import gc
from tqdm import tqdm

def get_video_info(video_path):
    """ffprobe를 사용하여 비디오 정보 추출"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,duration,r_frame_rate:format=duration',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info['streams'][0]
    width = int(stream['width'])
    height = int(stream['height'])
    nb_frames = int(stream.get('nb_frames', 0))
    
    if nb_frames == 0:
        duration = float(stream.get('duration', 0))
        if duration == 0 and 'format' in info:
            duration = float(info['format'].get('duration', 0))
        
        fps_str = stream.get('r_frame_rate', '0/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den) if float(den) != 0 else 0
        else:
            fps = float(fps_str)
        
        if duration > 0 and fps > 0:
            nb_frames = int(duration * fps)
    
    return width, height, nb_frames

def extract_y_frames(video_path, start_frame, max_frames):
    """
    비디오에서 Y 채널을 추출하여 텐서로 반환
    """
    width, height, total_frames = get_video_info(video_path)
    frames_to_load = min(total_frames - start_frame if total_frames > 0 else max_frames, max_frames)
    
    if frames_to_load <= 0:
        return None, width, height
    
    try:
        out, _ = (
            ffmpeg
            .input(video_path)
            .filter('select', f'gte(n,{start_frame})')
            .filter('setpts', 'N/(FR*TB)')
            .filter('extractplanes', 'y')
            .output('pipe:', format='rawvideo', pix_fmt='gray', vframes=frames_to_load)
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        return None, width, height
    
    frame_size = width * height
    num_frames = len(out) // frame_size
    
    if num_frames == 0:
        return None, width, height
    
    frames_np = np.frombuffer(out, np.uint8).reshape((num_frames, height, width)).copy()
    frames_tensor = torch.from_numpy(frames_np).float()
    
    del frames_np, out
    gc.collect()
    
    return frames_tensor, width, height

def detect_scenes_in_frequency_space(fft_shifted_frames, threshold, min_scene_frames):
    """
    주파수 공간에서 프레임 간 차이를 계산하여 scene 변화 감지
    
    Args:
        fft_shifted_frames: [N, H, W] complex tensor (FFT shifted)
        threshold: 차이 임계값 (높을수록 민감) - 정규화된 차이 비율
        min_scene_frames: 최소 scene 프레임 수 (240)
    
    Returns:
        scene_boundaries: scene 시작 프레임 인덱스 리스트
    """
    n_frames = fft_shifted_frames.shape[0]
    
    if n_frames <= 1:
        return [0]
    
    device = fft_shifted_frames.device
    
    # 주파수 공간에서 magnitude 계산
    magnitude = torch.abs(fft_shifted_frames)  # [N, H, W]
    
    # 저주파 영역만 사용 (중앙 영역 - 더 안정적인 scene 변화 감지)
    h, w = magnitude.shape[1], magnitude.shape[2]
    center_h, center_w = h // 2, w // 2
    crop_h, crop_w = h // 4, w // 4  # 중앙 25% 영역
    
    low_freq = magnitude[:, 
                        center_h - crop_h:center_h + crop_h,
                        center_w - crop_w:center_w + crop_w]
    
    # 연속된 프레임 간의 정규화된 차이 계산
    curr_frames = low_freq[:-1].reshape(n_frames - 1, -1)
    next_frames = low_freq[1:].reshape(n_frames - 1, -1)
    
    # L2 거리 계산 후 정규화
    diff = torch.norm(curr_frames - next_frames, p=2, dim=1)
    curr_norm = torch.norm(curr_frames, p=2, dim=1)
    
    # 상대적 변화율 계산 (0~1 사이 값)
    relative_changes = (diff / (curr_norm + 1e-8)).cpu().numpy()
    
    del curr_frames, next_frames, diff, curr_norm, magnitude, low_freq
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Scene 변화 감지
    scene_boundaries = [0]
    last_boundary = 0
    
    # 변화율이 threshold보다 크면 scene 변화로 감지
    for i in range(len(relative_changes)):
        if relative_changes[i] > threshold:
            frames_since_last = (i + 1) - last_boundary
            if frames_since_last >= min_scene_frames:
                scene_boundaries.append(i + 1)
                last_boundary = i + 1
    
    # 디버그 정보
    if len(relative_changes) > 0:
        print(f"    Change rate: min={relative_changes.min():.4f}, max={relative_changes.max():.4f}, "
              f"mean={relative_changes.mean():.4f}, std={relative_changes.std():.4f}")
        print(f"    Threshold: {threshold:.4f}, Changes above threshold: {(relative_changes > threshold).sum()}")
    
    return scene_boundaries

def save_frames_as_png(frames_tensor, output_dir, subdir_name):
    """
    프레임 텐서를 PNG 파일로 저장
    
    Args:
        frames_tensor: [N, H, W] tensor
        output_dir: 출력 디렉토리
        subdir_name: 서브디렉토리 이름 ('Y' 또는 'x')
    """
    channel_dir = os.path.join(output_dir, subdir_name)
    os.makedirs(channel_dir, exist_ok=True)
    
    num_frames = frames_tensor.shape[0]
    
    for i in range(num_frames):
        frame_np = frames_tensor[i].numpy().astype(np.uint8)
        img = Image.fromarray(frame_np, mode='L')
        
        frame_path = os.path.join(channel_dir, f"frame_{i+1:08d}.png")
        img.save(frame_path, compress_level=1)

def process_video_frequency_based(video_path, output_dir, video_name, device, 
                                  batch_size, similarity_threshold, min_scene_frames,
                                  crop_width, crop_height):
    """
    주파수 공간 기반 Scene 검출 및 Y 프레임 추출
    
    Args:
        video_path: 비디오 파일 경로
        output_dir: 출력 디렉토리
        video_name: 비디오 이름
        device: 처리 디바이스
        batch_size: 한 번에 처리할 프레임 수 (2400)
        similarity_threshold: scene 감지 임계값
        min_scene_frames: 최소 scene 프레임 수 (240)
        crop_width: 저주파 크롭 너비 (256)
        crop_height: 저주파 크롭 높이 (144)
    """
    width, height, total_frames = get_video_info(video_path)
    print(f"Video info: {width}x{height}, {total_frames} frames")
    
    current_frame = 0
    scene_idx = 1
    
    while current_frame < total_frames:
        print(f"\nProcessing frames {current_frame} to {min(current_frame + batch_size, total_frames)}...")
        
        # 1. 2400개의 Y 프레임을 GPU에 로드
        y_frames, _, _ = extract_y_frames(video_path, current_frame, batch_size)
        
        if y_frames is None or y_frames.shape[0] == 0:
            break
        
        y_frames = y_frames.to(device)
        num_frames = y_frames.shape[0]
        
        # 2. FFT 및 shift 수행
        print(f"  Performing FFT on {num_frames} frames...")
        fft_frames = torch.fft.fft2(y_frames)
        fft_shifted = torch.fft.fftshift(fft_frames, dim=(-2, -1))
        
        # 3. 주파수 공간에서 Scene 검출
        print(f"  Detecting scenes in frequency space...")
        scene_boundaries = detect_scenes_in_frequency_space(
            fft_shifted, 
            threshold=similarity_threshold,
            min_scene_frames=min_scene_frames
        )
        
        print(f"  Found {len(scene_boundaries)} scene(s) in this batch")
        
        # 4 & 5 & 6. 각 scene 처리
        for i in range(len(scene_boundaries)):
            scene_start_in_batch = scene_boundaries[i]
            
            if i + 1 < len(scene_boundaries):
                scene_end_in_batch = scene_boundaries[i + 1]
            else:
                scene_end_in_batch = num_frames
            
            global_start = current_frame + scene_start_in_batch
            global_end = current_frame + scene_end_in_batch
            
            scene_name = f"{video_name}_scene_{scene_idx:04d}"
            output_scene_dir = os.path.join(output_dir, scene_name)
            
            progress = (global_end / total_frames) * 100
            print(f"  Scene {scene_idx}: {global_end - global_start} frames | Progress: {progress:.1f}%")
            
            # 이 scene의 FFT 데이터 추출
            scene_fft = fft_shifted[scene_start_in_batch:scene_end_in_batch]
            
            # 5a. 원본 크기 역변환 (1280x720 원본 Y 이미지)
            scene_fft_full = torch.fft.ifftshift(scene_fft, dim=(-2, -1))
            scene_y_full = torch.fft.ifft2(scene_fft_full)
            scene_y_full = torch.real(scene_y_full).clamp(0, 255)
            
            # CPU로 이동하여 저장
            scene_y_full_cpu = scene_y_full.cpu()
            save_frames_as_png(scene_y_full_cpu, output_scene_dir, 'Y')
            del scene_y_full, scene_y_full_cpu
            
            # 4 & 5b. 저주파 이미지 생성 (올바른 방법)
            # 주파수 영역에서 중앙 crop_height x crop_width 영역만 추출
            center_y, center_x = height // 2, width // 2
            start_y = center_y - crop_height // 2
            end_y = start_y + crop_height
            start_x = center_x - crop_width // 2
            end_x = start_x + crop_width
            
            # 주파수 영역에서 직접 크롭 (shifted 상태에서)
            scene_fft_cropped = scene_fft[:, start_y:end_y, start_x:end_x]  # [N, crop_height, crop_width]
            
            # 크롭된 주파수 영역을 직접 IFFT (크기가 crop_height x crop_width로 유지됨)
            scene_fft_cropped_ishifted = torch.fft.ifftshift(scene_fft_cropped, dim=(-2, -1))
            scene_x_low = torch.fft.ifft2(scene_fft_cropped_ishifted)
            scene_x_low = torch.real(scene_x_low).clamp(0, 255)  # [N, crop_height, crop_width]
            
            # CPU로 이동하여 저장
            scene_x_low_cpu = scene_x_low.cpu()
            save_frames_as_png(scene_x_low_cpu, output_scene_dir, 'x')
            del scene_x_low, scene_x_low_cpu, scene_fft_cropped, scene_fft_cropped_ishifted
            
            scene_idx += 1
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # 메모리 해제
        del y_frames, fft_frames, fft_shifted
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # 다음 배치로 이동
        current_frame += num_frames
    
    print(f"\n✓ Total {scene_idx - 1} scenes extracted")

def YUV420_extractor(input_dir, output_dir, similarity_threshold, 
                     min_scene_frames, device, batch_size, 
                     crop_width, crop_height):
    """
    주파수 공간 기반 Y 프레임 추출 및 Scene 검출
    """
    print(f"Using device: {device}")
    
    if device == 'cuda':
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"GPU memory: {gpu_mem_allocated:.2f}GB / {gpu_mem_total:.2f}GB")
        torch.cuda.empty_cache()
    
    # 입력 디렉토리에서 모든 비디오 파일 찾기
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video(s)")
    
    for video_idx, video_path in enumerate(video_files, 1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n{'='*60}")
        print(f"[{video_idx}/{len(video_files)}] Processing: {video_name}")
        print(f"{'='*60}")
        
        try:
            process_video_frequency_based(
                video_path=video_path,
                output_dir=output_dir,
                video_name=video_name,
                device=device,
                batch_size=batch_size,
                similarity_threshold=similarity_threshold,
                min_scene_frames=min_scene_frames,
                crop_width=crop_width,
                crop_height=crop_height
            )
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and device == 'cuda':
                print(f"⚠ GPU out of memory! Skipping this video...")
                torch.cuda.empty_cache()
            else:
                print(f"Error processing video: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    # 최종 메모리 정리
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"\n{'='*60}")
    print(f"All videos processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    from config import INPUT_DIR, WORK_DIR, YUV420ExtractorConfig
    
    print("="*60)
    print("Frequency-based Scene Extractor (Y frames only)")
    print("="*60)
    print(f"Device: {YUV420ExtractorConfig.DEVICE}")
    print(f"Batch size: {YUV420ExtractorConfig.BATCH_SIZE} frames")
    print(f"Min scene length: {YUV420ExtractorConfig.MIN_SCENE_FRAMES} frames")
    print(f"Similarity threshold: {YUV420ExtractorConfig.SIMILARITY_THRESHOLD}")
    print(f"Low-freq crop size: {YUV420ExtractorConfig.CROP_WIDTH}x{YUV420ExtractorConfig.CROP_HEIGHT}")
    print("="*60)
    
    YUV420_extractor(
        input_dir=INPUT_DIR,
        output_dir=WORK_DIR,
        similarity_threshold=YUV420ExtractorConfig.SIMILARITY_THRESHOLD,
        min_scene_frames=YUV420ExtractorConfig.MIN_SCENE_FRAMES,
        device=YUV420ExtractorConfig.DEVICE,
        batch_size=YUV420ExtractorConfig.BATCH_SIZE,
        crop_width=YUV420ExtractorConfig.CROP_WIDTH,
        crop_height=YUV420ExtractorConfig.CROP_HEIGHT
    )
