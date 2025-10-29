# -*- coding: utf-8 -*-
"""
06_tensorRT_INT8.py - TensorRT INT8 추론으로 실제 이미지 처리 및 저장
"""

import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time
import glob
from PIL import Image
from FastSRGANconfig import fast_srgan_config

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT가 설치되지 않았습니다.")

class TensorRTCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 캘리브레이션을 위한 클래스"""
    
    def __init__(self, input_shape=(1, 1, 576, 1024), cache_file="tensorrt_calibration.cache"):
        super().__init__()
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.current_index = 0
        self.batch_size = input_shape[0]
        
        # 캘리브레이션 데이터 생성
        print("캘리브레이션 데이터 생성 중...")
        self.calibration_data = []
        for i in range(10):
            data = np.random.rand(*input_shape).astype(np.float32) * 0.5 + 0.25
            self.calibration_data.append(data)
        
        # GPU 메모리 할당
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_input = torch.zeros(input_shape, dtype=torch.float32, device=device)
        self.device_input_ptr = self.device_input.data_ptr()
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_index < len(self.calibration_data):
            batch = self.calibration_data[self.current_index]
            self.device_input.copy_(torch.from_numpy(batch).cuda())
            self.current_index += 1
            print(f"캘리브레이션 배치 {self.current_index}/{len(self.calibration_data)} 처리 중...")
            return [self.device_input_ptr]
        return None
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"기존 캘리브레이션 캐시 사용: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        print(f"캘리브레이션 캐시 저장: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_tensorrt_engine(onnx_path, engine_path):
    """ONNX 모델을 TensorRT INT8 엔진으로 변환"""
    if not TRT_AVAILABLE:
        return False
    
    # 절대 경로로 변환
    onnx_path = os.path.abspath(onnx_path)
    
    print(f"TensorRT INT8 엔진 빌드 시작")
    print(f"입력 ONNX: {onnx_path}")
    print(f"출력 엔진: {engine_path}")
    
    # ONNX 파일이 있는 디렉토리로 작업 디렉토리 변경 (외부 데이터 파일 접근용)
    original_cwd = os.getcwd()
    onnx_dir = os.path.dirname(onnx_path)
    if onnx_dir:
        os.chdir(onnx_dir)
        print(f"작업 디렉토리 변경: {onnx_dir}")
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    try:
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # ONNX 파싱
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("ONNX 파싱 실패:")
                for error in range(parser.num_errors):
                    print(f"  {parser.get_error(error)}")
                return False
        
        print("ONNX 모델 파싱 완료")
        
        # 빌더 설정
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
        # 동적 형태 설정
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_shape = input_tensor.shape
        
        min_shape = [1] + list(input_shape[1:])
        opt_shape = [1] + list(input_shape[1:])
        max_shape = [1] + list(input_shape[1:])
        
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # INT8 설정
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            calibrator = TensorRTCalibrator(
                input_shape=tuple(opt_shape),
                cache_file=os.path.join(original_cwd, "tensorrt_calibration.cache")
            )
            config.int8_calibrator = calibrator
            print("INT8 정밀도 활성화")
        else:
            print("INT8이 지원되지 않음")
            return False
        
        # 엔진 빌드
        print("TensorRT 엔진 빌드 중...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("엔진 빌드 실패")
            return False
        
        # 원래 디렉토리로 복귀
        os.chdir(original_cwd)
        
        # 엔진 저장
        os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        
        print(f"TensorRT 엔진 빌드 완료: {engine_path}")
        return True
        
    except Exception as e:
        print(f"엔진 빌드 중 오류: {e}")
        return False
    finally:
        # 항상 원래 디렉토리로 복귀
        os.chdir(original_cwd)

def load_image_as_tensor(image_path):
    """이미지를 텐서로 로드하고 전처리"""
    image = Image.open(image_path).convert('L')  # 그레이스케일로 변환
    image_array = np.array(image).astype(np.float32) / 255.0  # [0, 1] 범위로 정규화
    
    # (H, W) -> (1, 1, H, W) 형태로 변환
    tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    return tensor

def save_tensor_as_png(tensor, output_path):
    """텐서를 PNG 이미지로 저장"""
    # (1, 1, H, W) -> (H, W) 형태로 변환
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0).squeeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    
    # [0, 1] 범위를 [0, 255] 범위로 변환
    tensor = torch.clamp(tensor, 0, 1)
    image_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    
    # PIL Image로 변환하여 저장 (무손실 압축)
    image = Image.fromarray(image_array, mode='L')
    image.save(output_path, 'PNG', compress_level=6)  # 무손실 압축

def process_video_frames_with_tensorrt(work_dir, video_name, engine_path):
    """비디오 프레임들을 TensorRT INT8로 처리하여 저장"""
    if not TRT_AVAILABLE:
        print("TensorRT가 사용 불가능합니다.")
        return
    
    # 입력 및 출력 경로 설정
    input_dir = os.path.join(work_dir, video_name, "Y")
    output_dir = os.path.join(work_dir, video_name, "Y_fast_srgan_recon_int8")
    
    if not os.path.exists(input_dir):
        print(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 입력 프레임 목록 가져오기
    frame_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not frame_files:
        print(f"입력 프레임을 찾을 수 없습니다: {input_dir}")
        return
    
    print(f"총 {len(frame_files)}개의 프레임을 처리합니다.")
    print(f"입력 경로: {input_dir}")
    print(f"출력 경로: {output_dir}")
    
    # TensorRT 엔진 로드
    print(f"TensorRT 엔진 로드: {engine_path}")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("엔진 로드 실패")
        return
    
    context = engine.create_execution_context()
    
    # GPU 설정
    device = torch.device('cuda')
    
    # 입력/출력 텐서 이름 가져오기
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    # 프레임별 처리
    total_time = 0
    processed_count = 0
    
    for i, frame_path in enumerate(frame_files):
        try:
            # 이미지 로드
            input_tensor = load_image_as_tensor(frame_path).to(device)
            
            # 입력 형태 설정
            context.set_input_shape(input_name, input_tensor.shape)
            
            # 출력 크기 계산
            output_shape = context.get_tensor_shape(output_name)
            output_shape_tuple = tuple(output_shape)
            output_tensor = torch.zeros(output_shape_tuple, dtype=torch.float32, device=device)
            
            # 추론 실행
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            context.set_tensor_address(input_name, input_tensor.data_ptr())
            context.set_tensor_address(output_name, output_tensor.data_ptr())
            context.execute_async_v3(0)
            torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            total_time += inference_time
            
            # 결과 저장
            frame_name = os.path.basename(frame_path)
            output_path = os.path.join(output_dir, frame_name)
            save_tensor_as_png(output_tensor, output_path)
            
            processed_count += 1
            
            # 진행 상황 출력
            if (i + 1) % 50 == 0 or i == 0:
                avg_time = total_time / processed_count
                fps = 1.0 / avg_time
                print(f"처리 완료: {i+1}/{len(frame_files)} ({((i+1)/len(frame_files)*100):.1f}%), "
                      f"평균 시간: {avg_time*1000:.2f}ms, FPS: {fps:.2f}")
            
        except Exception as e:
            print(f"프레임 {frame_path} 처리 중 오류: {e}")
            continue
    
    # 최종 결과 출력
    if processed_count > 0:
        avg_time = total_time / processed_count
        fps = 1.0 / avg_time
        print(f"\n처리 완료!")
        print(f"총 처리된 프레임: {processed_count}/{len(frame_files)}")
        print(f"평균 추론 시간: {avg_time*1000:.2f} ms")
        print(f"평균 FPS: {fps:.2f}")
        print(f"출력 저장 경로: {output_dir}")
    else:
        print("처리된 프레임이 없습니다.")

def run_inference(engine_path, input_data):
    """TensorRT 엔진으로 추론 실행"""
    if not TRT_AVAILABLE:
        return None
    
    print(f"TensorRT 엔진 로드: {engine_path}")
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("엔진 로드 실패")
        return None
    
    context = engine.create_execution_context()
    
    # GPU 메모리 할당
    device = torch.device('cuda')
    input_tensor = torch.from_numpy(input_data).float().to(device)
    
    # 입력/출력 텐서 이름 가져오기
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    # 입력 형태 설정
    context.set_input_shape(input_name, input_data.shape)
    
    # 출력 크기 계산
    output_shape = context.get_tensor_shape(output_name)
    output_shape_tuple = tuple(output_shape)  # TensorRT Dims를 튜플로 변환
    output_tensor = torch.zeros(output_shape_tuple, dtype=torch.float32, device=device)
    
    print(f"입력 형태: {input_data.shape}")
    print(f"출력 형태: {output_shape_tuple}")
    
    # 성능 측정
    warmup_runs = 10
    test_runs = 100
    
    print(f"워밍업 {warmup_runs}회...")
    for _ in range(warmup_runs):
        context.set_tensor_address(input_name, input_tensor.data_ptr())
        context.set_tensor_address(output_name, output_tensor.data_ptr())
        context.execute_async_v3(0)  # 스트림 핸들 0 사용
        torch.cuda.synchronize()
    
    print(f"성능 측정 {test_runs}회...")
    times = []
    
    for _ in range(test_runs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        context.set_tensor_address(input_name, input_tensor.data_ptr())
        context.set_tensor_address(output_name, output_tensor.data_ptr())
        context.execute_async_v3(0)
        torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"평균 추론 시간: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    
    return avg_time

def find_onnx_model():
    """원본 ONNX 모델 찾기"""
    available_models = []
    
    for search_dir in ['onnx_models', '.']:
        if os.path.exists(search_dir):
            models = glob.glob(os.path.join(search_dir, "*.onnx"))
            available_models.extend(models)
    
    # 원본 모델 우선 선택
    for model in available_models:
        if 'int8' not in model and 'quantized' not in model:
            return model
    
    return available_models[0] if available_models else None

def main():
    print("TensorRT INT8 비디오 프레임 처리")
    print("=" * 60)
    
    # ONNX 모델 찾기
    onnx_path = find_onnx_model()
    if not onnx_path:
        print("ONNX 모델을 찾을 수 없습니다.")
        return
    
    print(f"원본 모델 선택: {onnx_path}")
    
    # 엔진 경로 설정
    model_name = Path(onnx_path).stem
    engine_path = f"tensorrt_engines/{model_name}_int8.trt"
    
    # 엔진 빌드
    if not os.path.exists(engine_path):
        print("\nTensorRT 엔진 빌드 중...")
        if not build_tensorrt_engine(onnx_path, engine_path):
            print("엔진 빌드 실패")
            return
    else:
        print(f"\n기존 엔진 사용: {engine_path}")
    
    # work_dir에서 비디오 폴더 찾기
    work_dir = fast_srgan_config.work_dir
    if not os.path.exists(work_dir):
        print(f"작업 디렉토리를 찾을 수 없습니다: {work_dir}")
        return
    
    # 비디오 폴더 목록 가져오기
    video_folders = [d for d in os.listdir(work_dir) 
                    if os.path.isdir(os.path.join(work_dir, d)) and 
                    os.path.exists(os.path.join(work_dir, d, "Y"))]
    
    if not video_folders:
        print(f"처리할 비디오 폴더를 찾을 수 없습니다: {work_dir}")
        return
    
    print(f"\n발견된 비디오 폴더: {video_folders}")
    
    # 각 비디오 폴더 처리
    for video_name in video_folders:
        print(f"\n비디오 '{video_name}' 처리 시작...")
        process_video_frames_with_tensorrt(work_dir, video_name, engine_path)
        print(f"비디오 '{video_name}' 처리 완료!")

if __name__ == "__main__":
    main()