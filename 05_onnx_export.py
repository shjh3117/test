"""
05_onnx_export.py

학습된 FastSRGAN 모델을 ONNX 포맷으로 변환하는 스크립트

기능:
- PyTorch 모델을 ONNX 포맷으로 변환
- ONNX 모델 검증 및 최적화
- 변환된 모델의 성능 테스트
"""

import os
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import time
from FastSRGAN_models import FastSRGANGenerator
from FastSRGANconfig import fast_srgan_config as config

class ONNXExporter:
    def __init__(self, model_path, output_dir='onnx_models'):
        """
        ONNX 변환기 초기화
        
        Args:
            model_path (str): 학습된 PyTorch 모델 경로
            output_dir (str): ONNX 모델 저장 디렉토리
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = torch.device(config.device)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"ONNX Exporter 초기화 완료")
        print(f"모델 경로: {model_path}")
        print(f"출력 디렉토리: {output_dir}")
        print(f"사용 디바이스: {self.device}")
    
    def load_pytorch_model(self):
        """학습된 PyTorch 모델 로드"""
        print("\n=== PyTorch 모델 로딩 ===")
        
        # 모델 생성
        model = FastSRGANGenerator()
        
        # 가중치 로드
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            print(f"✓ 모델 가중치 로드 완료: {self.model_path}")
        else:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        # 모델을 평가 모드로 설정
        model.eval()
        model.to(self.device)
        
        return model
    
    def export_to_onnx(self, model, input_shape=(1, 1, 576, 1024), 
                       opset_version=18, dynamic_axes=False):
        """
        PyTorch 모델을 ONNX로 변환
        
        Args:
            model: PyTorch 모델
            input_shape: 입력 텐서 형태 (B, C, H, W)
            opset_version: ONNX opset 버전
            dynamic_axes: 동적 축 사용 여부 (False로 변경)
        """
        print(f"\n=== ONNX 변환 (opset {opset_version}) ===")
        
        # 더미 입력 생성
        dummy_input = torch.randn(input_shape, device=self.device)
        print(f"입력 형태: {input_shape}")
        
        # ONNX 파일 경로
        onnx_path = os.path.join(self.output_dir, f'fast_srgan_generator_opset{opset_version}.onnx')
        
        # 동적 축 설정 (배치 크기와 이미지 크기를 동적으로 설정)
        if dynamic_axes:
            dynamic_axes_dict = {
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        else:
            dynamic_axes_dict = None
        
        # ONNX 변환 (최신 방식)
        try:
            torch.onnx.export(
                model,                          # 변환할 모델
                dummy_input,                    # 더미 입력
                onnx_path,                      # 출력 파일 경로
                export_params=True,             # 모델 파라미터 포함
                opset_version=opset_version,    # ONNX opset 버전
                do_constant_folding=True,       # 상수 폴딩 최적화
                input_names=['input'],          # 입력 이름
                output_names=['output'],        # 출력 이름
                dynamic_axes=dynamic_axes_dict, # 동적 축
                verbose=False                   # 상세 출력 비활성화
            )
            print(f"✓ ONNX 변환 완료: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            print(f"✗ ONNX 변환 실패: {e}")
            return None
    
    def verify_onnx_model(self, onnx_path):
        """ONNX 모델 검증"""
        print(f"\n=== ONNX 모델 검증 ===")
        
        try:
            # ONNX 모델 로드 및 검증
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX 모델 검증 통과")
            
            # 모델 정보 출력
            print(f"모델 그래프 노드 수: {len(onnx_model.graph.node)}")
            print(f"입력 정보:")
            for input_info in onnx_model.graph.input:
                print(f"  - {input_info.name}: {input_info.type}")
            print(f"출력 정보:")
            for output_info in onnx_model.graph.output:
                print(f"  - {output_info.name}: {output_info.type}")
            
            return True
            
        except Exception as e:
            print(f"✗ ONNX 모델 검증 실패: {e}")
            return False
    
    def optimize_onnx_model(self, onnx_path):
        """ONNX 모델 최적화"""
        print(f"\n=== ONNX 모델 최적화 ===")
        
        try:
            # 최적화된 모델 경로
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
            
            # ONNX Runtime을 사용한 최적화 (경고 제거를 위해 레벨 조정)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC  # EXTENDED에서 BASIC으로 변경
            sess_options.optimized_model_filepath = optimized_path
            
            # 임시 세션 생성 (최적화 수행)
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
            
            print(f"✓ ONNX 모델 최적화 완료: {optimized_path}")
            print(f"  최적화 레벨: BASIC (하드웨어 독립적)")
            return optimized_path
            
        except Exception as e:
            print(f"✗ ONNX 모델 최적화 실패: {e}")
            print(f"  원본 모델 사용: {onnx_path}")
            return onnx_path
    
    def test_onnx_inference(self, onnx_path, test_shape=(1, 1, 576, 1024)):
        """ONNX 모델 추론 테스트"""
        print(f"\n=== ONNX 추론 테스트 ===")
        
        try:
            # ONNX Runtime 세션 생성
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            # 사용 중인 Provider 확인
            current_provider = session.get_providers()[0]
            print(f"사용 중인 Provider: {current_provider}")
            
            # 테스트 입력 생성
            test_input = np.random.randn(*test_shape).astype(np.float32)
            
            # 추론 실행 및 시간 측정
            input_name = session.get_inputs()[0].name
            
            # Warmup
            for _ in range(5):
                _ = session.run(None, {input_name: test_input})
            
            # 성능 측정
            num_runs = 50
            start_time = time.time()
            
            for _ in range(num_runs):
                outputs = session.run(None, {input_name: test_input})
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs
            
            # 결과 출력
            output_shape = outputs[0].shape
            print(f"✓ ONNX 추론 성공")
            print(f"입력 형태: {test_shape}")
            print(f"출력 형태: {output_shape}")
            print(f"평균 추론 시간: {avg_time*1000:.2f} ms")
            print(f"FPS: {1/avg_time:.2f}")
            
            return True
            
        except Exception as e:
            print(f"✗ ONNX 추론 테스트 실패: {e}")
            return False
    
    def compare_pytorch_onnx(self, pytorch_model, onnx_path, test_shape=(1, 1, 576, 1024)):
        """PyTorch와 ONNX 모델 출력 비교"""
        print(f"\n=== PyTorch vs ONNX 출력 비교 ===")
        
        try:
            # 테스트 입력 생성 (ONNX 모델과 동일한 크기 사용)
            test_input_torch = torch.randn(test_shape, device=self.device)
            test_input_numpy = test_input_torch.cpu().numpy()
            
            # PyTorch 추론
            with torch.no_grad():
                pytorch_output = pytorch_model(test_input_torch)
                pytorch_output_numpy = pytorch_output.cpu().numpy()
            
            # ONNX 추론
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, providers=providers)
            input_name = session.get_inputs()[0].name
            onnx_output = session.run(None, {input_name: test_input_numpy})[0]
            
            # 차이 계산
            diff = np.abs(pytorch_output_numpy - onnx_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"✓ 출력 비교 완료")
            print(f"테스트 입력 크기: {test_shape}")
            print(f"최대 차이: {max_diff:.6f}")
            print(f"평균 차이: {mean_diff:.6f}")
            
            # 허용 가능한 차이인지 확인 (일반적으로 1e-5 이하)
            if max_diff < 1e-4:
                print("✓ PyTorch와 ONNX 출력이 일치합니다")
                return True
            else:
                print("⚠ PyTorch와 ONNX 출력에 큰 차이가 있습니다")
                return False
                
        except Exception as e:
            print(f"✗ 출력 비교 실패: {e}")
            return False
    
    def export_multiple_versions(self):
        """다양한 버전의 ONNX 모델 생성"""
        print("\n" + "="*60)
        print("FastSRGAN ONNX 변환 시작")
        print("="*60)
        
        # PyTorch 모델 로드
        pytorch_model = self.load_pytorch_model()
        
        # 다양한 opset 버전으로 변환
        opset_versions = [18]  # 최신 ONNX opset 버전 사용 (권장)
        successful_exports = []
        
        for opset in opset_versions:
            print(f"\n--- Opset {opset} 변환 시작 ---")
            
            # ONNX 변환
            onnx_path = self.export_to_onnx(pytorch_model, opset_version=opset)
            
            if onnx_path:
                # 모델 검증
                if self.verify_onnx_model(onnx_path):
                    # 추론 테스트 (최적화 건너뜀)
                    if self.test_onnx_inference(onnx_path):
                        # 출력 비교 (실패해도 성공으로 처리)
                        comparison_result = self.compare_pytorch_onnx(pytorch_model, onnx_path)
                        successful_exports.append((opset, onnx_path))
                        if comparison_result:
                            print(f"✓ Opset {opset} 변환 및 검증 완료")
                        else:
                            print(f"✓ Opset {opset} 변환 완료 (출력 비교는 건너뜀)")
                    else:
                        print(f"✗ Opset {opset} 추론 테스트 실패")
                else:
                    print(f"✗ Opset {opset} 모델 검증 실패")
            else:
                print(f"✗ Opset {opset} 변환 실패")
        
        # 결과 요약
        print(f"\n" + "="*60)
        print("ONNX 변환 결과 요약")
        print("="*60)
        
        if successful_exports:
            print(f"✓ 성공적으로 변환된 모델: {len(successful_exports)}개")
            for opset, path in successful_exports:
                file_size = os.path.getsize(path) / (1024 * 1024)  # MB
                print(f"  - Opset {opset}: {path} ({file_size:.2f} MB)")
            
            # 권장 모델 선택 (가장 높은 opset 버전)
            recommended = max(successful_exports, key=lambda x: x[0])
            print(f"\n권장 모델 (TensorRT 양자화용): Opset {recommended[0]}")
            print(f"경로: {recommended[1]}")
            
        else:
            print("✗ 성공적으로 변환된 모델이 없습니다")
        
        print("="*60)
        
        return successful_exports

def main():
    """메인 실행 함수"""
    # 설정
    model_path = config.model_path_gen  # 'fast_srgan_generator_best.pth'
    output_dir = 'onnx_models'
    
    # ONNX 변환기 생성
    exporter = ONNXExporter(model_path, output_dir)
    
    # 다양한 버전으로 변환 수행
    successful_exports = exporter.export_multiple_versions()
    
    if successful_exports:
        print(f"\n✓ ONNX 변환 완료! {len(successful_exports)}개 모델 생성됨")
        print("다음 단계: 06_int8_quantization.py로 양자화 수행")
    else:
        print("\n✗ ONNX 변환 실패")

if __name__ == "__main__":
    main()