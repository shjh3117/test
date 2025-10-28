"""
input_dir아래의 모든 영상들은 항상 yuv420영상이다.
프레임 별로 Y, U, V를 추출하여 그레이스케일 무손실 압축(png)로 저장한다.
ex)
work_dir/input/Y/frame_00000001.png
work_dir/input2/Y/frame_00000001.png
"""

import os
import glob
import ffmpeg

def YUV420_extractor(input_dir = 'videos', output_dir = 'work_dir'):
    """
    YUV420 비디오에서 Y, U, V 채널을 프레임별로 추출하여 PNG로 저장
    """
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
        print(f"Processing video: {video_name}")
        
        # 출력 디렉토리 생성
        output_video_dir = os.path.join(output_dir, video_name)
        y_dir = os.path.join(output_video_dir, 'Y')
        u_dir = os.path.join(output_video_dir, 'U')
        v_dir = os.path.join(output_video_dir, 'V')
        
        os.makedirs(y_dir, exist_ok=True)
        os.makedirs(u_dir, exist_ok=True)
        os.makedirs(v_dir, exist_ok=True)
        
        try:
            # Y 채널 추출 (그레이스케일)
            print("Extracting Y channel...")
            (
                ffmpeg
                .input(video_path)
                .filter('extractplanes', 'y')
                .output(os.path.join(y_dir, 'frame_%08d.png'), pix_fmt='gray', compression_level=0)
                .overwrite_output()
                .run(quiet=True)
            )
            
            # U 채널 추출 (그레이스케일)
            print("Extracting U channel...")
            (
                ffmpeg
                .input(video_path)
                .filter('extractplanes', 'u')
                .output(os.path.join(u_dir, 'frame_%08d.png'), pix_fmt='gray', compression_level=0)
                .overwrite_output()
                .run(quiet=True)
            )
            
            # V 채널 추출 (그레이스케일)
            print("Extracting V channel...")
            (
                ffmpeg
                .input(video_path)
                .filter('extractplanes', 'v')
                .output(os.path.join(v_dir, 'frame_%08d.png'), pix_fmt='gray', compression_level=0)
                .overwrite_output()
                .run(quiet=True)
            )
            
            print(f"Successfully extracted Y, U, V channels from {video_name}")
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            continue

if __name__ == "__main__":
    YUV420_extractor()
