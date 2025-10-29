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

def extract_y_frames_for_scene_detection(video_path, device='cuda', max_frames=240, start_frame=0):
    """
    비디오에서 Y 채널을 추출하여 scene 감지용 텐서 생성 (임시 파일 없이)
    
    Args:
        video_path: 비디오 파일 경로
        device: 'cuda' 또는 'cpu'
        max_frames: 최대 로드할 프레임 수 (scene 감지용, 240 권장)
        start_frame: 시작 프레임 인덱스
    
    Returns:
        y_frames: [N, H, W] tensor (GPU/CPU), N은 최대 max_frames
        total_frames: 비디오의 실제 총 프레임 수
    """
    # 비디오 정보 가져오기
    width, height, total_frames = get_video_info(video_path)
    
    if total_frames == 0:
        # nb_frames를 얻지 못한 경우 직접 카운트
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream:
                total_frames = int(video_stream.get('nb_frames', 0))
        except:
            total_frames = max_frames  # 폴백
    
    frames_to_load = min(total_frames - start_frame if total_frames > 0 else max_frames, max_frames)
    
    if frames_to_load <= 0:
        return None, total_frames if total_frames > 0 else 0
    
    print(f"Loading {frames_to_load} frames starting from frame {start_frame}...")
    
    # ffmpeg로 Y 채널을 numpy array로 직접 읽기
    try:
        # select 필터 사용: gte로 시작점 지정, 프레임 수 제한은 vframes로
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
        raise
    
    # 프레임 크기 계산 (Y 채널만)
    frame_size = width * height
    num_frames = len(out) // frame_size
    
    if num_frames == 0:
        return None, total_frames if total_frames > 0 else 0
    
    # numpy array로 변환 (copy()로 writable하게 만들기)
    frames_np = np.frombuffer(out, np.uint8).reshape((num_frames, height, width)).copy()
    
    # torch tensor로 변환
    frames_tensor = torch.from_numpy(frames_np).float().to(device)
    
    del frames_np, out
    gc.collect()
    
    print(f"Loaded {num_frames} frames to {device}")
    
    return frames_tensor, total_frames if total_frames > 0 else num_frames

def detect_scene_changes(y_frames, threshold=0.85, min_scene_frames=8, hard_threshold=0.3):
    """
    Y 프레임 간의 정규화된 내적을 계산하여 scene 변화 감지
    
    Args:
        y_frames: [N, H, W] tensor
        threshold: 유사도 임계값 (이보다 낮으면 scene 변화로 감지)
        min_scene_frames: 최소 scene 프레임 수 (기본 8)
        hard_threshold: 강제 scene 분리 임계값 (이보다 낮으면 무조건 분리, 기본 0.3)
    
    Returns:
        scene_boundaries: scene 시작 프레임 인덱스 리스트
        similarities: 연속 프레임 간 유사도 배열
    """
    n_frames = y_frames.shape[0]
    
    if n_frames <= 1:
        return [0], []
    
    device = y_frames.device
    
    # 연속된 프레임 간의 코사인 유사도 계산
    curr_frames = y_frames[:-1].reshape(n_frames - 1, -1)
    next_frames = y_frames[1:].reshape(n_frames - 1, -1)
    
    # 정규화
    curr_norm = torch.nn.functional.normalize(curr_frames, p=2, dim=1)
    next_norm = torch.nn.functional.normalize(next_frames, p=2, dim=1)
    
    # 유사도 계산
    similarities = torch.sum(curr_norm * next_norm, dim=1).cpu().numpy()
    
    # 메모리 해제
    del curr_frames, next_frames, curr_norm, next_norm
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # scene 변화 감지 (2단계 임계값)
    # hard_threshold 이하: 무조건 scene 분리
    # threshold 이하: 최소 프레임 조건 확인
    hard_changes = (similarities < hard_threshold)
    soft_changes = (similarities < threshold)
    
    # scene 시작 인덱스 구성
    scene_boundaries = [0]
    last_boundary = 0
    
    for i in range(len(similarities)):
        if hard_changes[i]:
            # 강제 분리 (유사도가 매우 낮음)
            scene_boundaries.append(i + 1)
            last_boundary = i + 1
            print(f"  Hard scene change at frame {i+1} (similarity: {similarities[i]:.3f})")
        elif soft_changes[i]:
            # 일반 분리 (최소 프레임 수 확인)
            frames_since_last = (i + 1) - last_boundary
            if frames_since_last >= min_scene_frames:
                scene_boundaries.append(i + 1)
                last_boundary = i + 1
                print(f"  Scene change at frame {i+1} (similarity: {similarities[i]:.3f}, length: {frames_since_last})")
            else:
                print(f"  Skipped scene change at frame {i+1} (similarity: {similarities[i]:.3f}, too short: {frames_since_last} < {min_scene_frames})")
    
    return scene_boundaries, similarities

def save_y_frames_as_png(y_frames_tensor, output_dir, channel='Y'):
    """
    Y 프레임 텐서를 PNG 파일로 저장
    
    Args:
        y_frames_tensor: [N, H, W] CPU tensor
        output_dir: 출력 디렉토리
        channel: 채널 이름 ('Y', 'U', 'V')
    """
    channel_dir = os.path.join(output_dir, channel)
    os.makedirs(channel_dir, exist_ok=True)
    
    num_frames = y_frames_tensor.shape[0]
    
    for i in range(num_frames):
        frame_np = y_frames_tensor[i].numpy().astype(np.uint8)
        img = Image.fromarray(frame_np, mode='L')
        
        frame_path = os.path.join(channel_dir, f"frame_{i+1:08d}.png")
        img.save(frame_path, compress_level=1)
    
    print(f"  ✓ Saved {num_frames} {channel} frames to {channel_dir}")

def extract_uv_frames(video_path, start_frame, end_frame, output_dir, width, height):
    """
    특정 구간의 U, V 채널을 추출하여 PNG로 저장
    
    Args:
        video_path: 비디오 파일 경로
        start_frame: 시작 프레임 (0-based)
        end_frame: 끝 프레임 (exclusive)
        output_dir: 출력 디렉토리
        width: 비디오 너비
        height: 비디오 높이
    """
    num_frames = end_frame - start_frame
    
    print(f"  Extracting U/V channels for {num_frames} frames (frame {start_frame} to {end_frame-1})...")
    
    # U 채널 추출
    print(f"  Extracting U channel...")
    try:
        out_u, _ = (
            ffmpeg
            .input(video_path)
            .filter('select', f'gte(n,{start_frame})')
            .filter('setpts', 'N/(FR*TB)')
            .filter('extractplanes', 'u')
            .output('pipe:', format='rawvideo', pix_fmt='gray', vframes=num_frames)
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        
        # U 채널 저장 (YUV420이므로 크기가 절반)
        u_width = width // 2
        u_height = height // 2
        frame_size_u = u_width * u_height
        u_frames = np.frombuffer(out_u, np.uint8).reshape((-1, u_height, u_width)).copy()
        
        u_dir = os.path.join(output_dir, 'U')
        os.makedirs(u_dir, exist_ok=True)
        
        for i in range(u_frames.shape[0]):
            img = Image.fromarray(u_frames[i], mode='L')
            img.save(os.path.join(u_dir, f"frame_{i+1:08d}.png"), compress_level=1)
        
        print(f"  ✓ Saved {u_frames.shape[0]} U frames")
        del out_u, u_frames
        
    except Exception as e:
        print(f"  ⚠ Error extracting U channel: {e}")
    
    # V 채널 추출
    print(f"  Extracting V channel...")
    try:
        out_v, _ = (
            ffmpeg
            .input(video_path)
            .filter('select', f'gte(n,{start_frame})')
            .filter('setpts', 'N/(FR*TB)')
            .filter('extractplanes', 'v')
            .output('pipe:', format='rawvideo', pix_fmt='gray', vframes=num_frames)
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        
        # V 채널 저장
        v_frames = np.frombuffer(out_v, np.uint8).reshape((-1, u_height, u_width)).copy()
        
        v_dir = os.path.join(output_dir, 'V')
        os.makedirs(v_dir, exist_ok=True)
        
        for i in range(v_frames.shape[0]):
            img = Image.fromarray(v_frames[i], mode='L')
            img.save(os.path.join(v_dir, f"frame_{i+1:08d}.png"), compress_level=1)
        
        print(f"  ✓ Saved {v_frames.shape[0]} V frames")
        del out_v, v_frames
        
    except Exception as e:
        print(f"  ⚠ Error extracting V channel: {e}")
    
    gc.collect()

def process_video_with_chunked_detection(video_path, output_dir, video_name, device='cuda', 
                                        chunk_size=240, similarity_threshold=0.85, 
                                        min_scene_frames=8, hard_threshold=0.3):
    """
    240프레임씩 Y 채널을 로드하여 scene 감지 후, 감지된 구간의 Y/U/V를 추출하는 통합 함수
    
    처리 순서:
    1. 240개의 Y프레임 추출 → GPU 이동
    2. GPU에서 scene 검사 (예: 51번째에서 감지)
    3. 1-50 프레임을 GPU→CPU 이동하여 scene1에 Y 저장
    4. 해당 구간의 U/V도 추출하여 저장
    5. 51-240 프레임으로 다시 검사
    6. 241-480 프레임 로드하여 반복
    
    Args:
        video_path: 비디오 파일 경로
        output_dir: 출력 디렉토리
        video_name: 비디오 이름
        device: 처리 디바이스
        chunk_size: 청크 크기 (240)
        similarity_threshold: scene 변화 감지 임계값
        min_scene_frames: 최소 scene 프레임 수 (기본 8)
        hard_threshold: 강제 scene 분리 임계값 (기본 0.3)
    """
    # 총 프레임 수 확인
    width, height, total_frames = get_video_info(video_path)
    print(f"Video info: {width}x{height}, {total_frames} frames")
    
    current_frame = 0
    scene_idx = 1
    
    while current_frame < total_frames:
        chunk_end = min(current_frame + chunk_size, total_frames)
        chunk_length = chunk_end - current_frame
        
        print(f"\n{'='*60}")
        print(f"Processing chunk: frames {current_frame}-{chunk_end-1} ({chunk_length} frames)")
        print(f"{'='*60}")
        
        try:
            # 1. 240개의 Y프레임 추출
            print(f"Step 1: Extracting {chunk_length} Y frames...")
            y_frames, _ = extract_y_frames_for_scene_detection(
                video_path, device='cpu', max_frames=chunk_length, start_frame=current_frame
            )
            
            if y_frames is None or y_frames.shape[0] == 0:
                print(f"Warning: No frames in chunk starting at {current_frame}")
                break
            
            # 2. GPU로 이동
            print(f"Step 2: Moving {y_frames.shape[0]} frames to {device}...")
            y_frames = y_frames.to(device)
            
            # 3. GPU에서 Scene 검사
            print(f"Step 3: Detecting scenes on {device}...")
            scene_boundaries, _ = detect_scene_changes(
                y_frames, 
                threshold=similarity_threshold,
                min_scene_frames=min_scene_frames,
                hard_threshold=hard_threshold
            )
            
            print(f"Detected boundaries within chunk: {scene_boundaries}")
            
            # 4-7. 감지된 scene별로 처리
            if len(scene_boundaries) == 1:
                # scene 변화가 없음 → 전체 240프레임이 하나의 scene
                print(f"No scene change detected - entire chunk is scene {scene_idx}")
                
                scene_start_in_chunk = 0
                scene_end_in_chunk = y_frames.shape[0]
                global_start = current_frame
                global_end = current_frame + scene_end_in_chunk
                
                # Y 프레임 저장
                scene_name = f"{video_name}_scene_{scene_idx:04d}"
                output_scene_dir = os.path.join(output_dir, scene_name)
                print(f"Saving scene {scene_idx}: frames {global_start}-{global_end-1}")
                
                # 5. GPU → CPU 이동
                y_frames_cpu = y_frames[scene_start_in_chunk:scene_end_in_chunk].cpu()
                
                # 6. Y 프레임 저장
                save_y_frames_as_png(y_frames_cpu, output_scene_dir, 'Y')
                del y_frames_cpu
                
                # 7. U, V 프레임 추출 및 저장
                extract_uv_frames(video_path, global_start, global_end, output_scene_dir, width, height)
                
                scene_idx += 1
                
            else:
                # scene 변화가 감지됨 → 여러 scene으로 분할
                for i in range(len(scene_boundaries)):
                    scene_start_in_chunk = scene_boundaries[i]
                    
                    # 다음 boundary까지가 현재 scene
                    if i + 1 < len(scene_boundaries):
                        scene_end_in_chunk = scene_boundaries[i + 1]
                    else:
                        scene_end_in_chunk = y_frames.shape[0]
                    
                    global_start = current_frame + scene_start_in_chunk
                    global_end = current_frame + scene_end_in_chunk
                    
                    scene_name = f"{video_name}_scene_{scene_idx:04d}"
                    output_scene_dir = os.path.join(output_dir, scene_name)
                    
                    print(f"\nProcessing detected scene {scene_idx}: frames {global_start}-{global_end-1} ({global_end - global_start} frames)")
                    
                    # 5. GPU → CPU 이동
                    y_frames_cpu = y_frames[scene_start_in_chunk:scene_end_in_chunk].cpu()
                    
                    # 6. Y 프레임 저장
                    save_y_frames_as_png(y_frames_cpu, output_scene_dir, 'Y')
                    del y_frames_cpu
                    gc.collect()
                    
                    # 7. U, V 프레임 추출 및 저장
                    extract_uv_frames(video_path, global_start, global_end, output_scene_dir, width, height)
                    
                    scene_idx += 1
            
            # 메모리 해제
            del y_frames
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing chunk at frame {current_frame}: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # 8. 다음 청크로 이동 (241번째 프레임부터)
        current_frame = chunk_end
    
    print(f"\n{'='*60}")
    print(f"✓ Completed processing: {scene_idx - 1} scenes extracted")
    print(f"{'='*60}")

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
                     min_scene_frames=8, hard_threshold=0.3, force_cpu=False):
    """
    YUV420 비디오에서 Y 프레임 유사성 기반으로 scene을 감지하고,
    scene별로 Y, U, V 채널을 추출하여 PNG로 저장
    
    Args:
        input_dir: 입력 비디오 디렉토리
        output_dir: 출력 디렉토리
        similarity_threshold: scene 감지 임계값 (낮을수록 민감)
        min_scene_frames: 최소 scene 프레임 수 (기본 8)
        hard_threshold: 강제 scene 분리 임계값 (기본 0.3)
        force_cpu: True이면 GPU 사용 안함 (메모리 부족 시)
    """
    if force_cpu:
        device = 'cpu'
        print("Forced CPU mode")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    if device == 'cuda':
        # GPU 메모리 상태 출력
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"GPU memory: {gpu_mem_allocated:.2f}GB / {gpu_mem_total:.2f}GB")
        
        # GPU 메모리 초기화
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
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
        
        try:
            # 새로운 통합 처리 함수 호출
            process_video_with_chunked_detection(
                video_path=video_path,
                output_dir=output_dir,
                video_name=video_name,
                device=device,
                chunk_size=240,
                similarity_threshold=similarity_threshold,
                min_scene_frames=min_scene_frames,
                hard_threshold=hard_threshold
            )
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and device == 'cuda':
                print(f"⚠ GPU out of memory! Retrying with CPU...")
                torch.cuda.empty_cache()
                process_video_with_chunked_detection(
                    video_path=video_path,
                    output_dir=output_dir,
                    video_name=video_name,
                    device='cpu',
                    chunk_size=240,
                    similarity_threshold=similarity_threshold,
                    min_scene_frames=min_scene_frames,
                    hard_threshold=hard_threshold
                )
            else:
                raise
                
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 최종 메모리 정리
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"\n{'='*60}")
    print(f"All videos processed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YUV420 Video Scene Extractor')
    parser.add_argument('--input_dir', default='videos', help='Input video directory')
    parser.add_argument('--output_dir', default='work_dir', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.9, help='Scene detection threshold (soft)')
    parser.add_argument('--hard_threshold', type=float, default=0.45, help='Hard scene detection threshold (force split)')
    parser.add_argument('--min_frames', type=int, default=8, help='Minimum frames per scene')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode (if GPU memory is limited)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("YUV420 Scene Extractor")
    print("="*60)
    print("Processing 240 frames at a time")
    print("Max scene length: 240 frames")
    print(f"Min scene length: {args.min_frames} frames (except hard threshold)")
    print(f"Soft threshold: {args.threshold}, Hard threshold: {args.hard_threshold}")
    print("="*60)
    
    YUV420_extractor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        similarity_threshold=args.threshold,
        min_scene_frames=args.min_frames,
        hard_threshold=args.hard_threshold,
        force_cpu=args.cpu
    )
