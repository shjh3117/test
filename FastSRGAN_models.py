"""
Fast-SRGAN (AnimeSRGAN) 모델 정의
애니메이션 고주파 복원에 최적화된 경량 SRGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from FastSRGANconfig import fast_srgan_config as config

class ChannelAttention(nn.Module):
    """Channel Attention Module - 고주파 특성 강화"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module - 엣지 및 고주파 영역 강화"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class MultiScaleConv(nn.Module):
    """Multi-Scale Convolution - 다양한 크기의 고주파 패턴 감지"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        
        # 다양한 kernel size로 고주파 패턴 감지
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels // 4, 1, padding=0)
        self.conv_3x3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.conv_5x5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.conv_7x7 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out1 = self.conv_1x1(x)
        out2 = self.conv_3x3(x)
        out3 = self.conv_5x5(x)
        out4 = self.conv_7x7(x)
        
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        
        return out

class DilatedConvBlock(nn.Module):
    """Dilated Convolution Block - 넓은 수용 영역으로 컨텍스트 파악"""
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4]):
        super(DilatedConvBlock, self).__init__()
        
        self.dilated_convs = nn.ModuleList()
        for dilation in dilation_rates:
            self.dilated_convs.append(
                nn.Conv2d(in_channels, out_channels // len(dilation_rates), 
                         3, padding=dilation, dilation=dilation)
            )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        outputs = []
        for conv in self.dilated_convs:
            outputs.append(conv(x))
        
        out = torch.cat(outputs, dim=1)
        out = self.bn(out)
        out = self.relu(out)
        
        return out

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution - MobileNet의 핵심 구성요소"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, 
            groups=in_channels, bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

class InvertedResidualBlock(nn.Module):
    """MobileNetV2의 Inverted Residual Block"""
    def __init__(self, in_channels, out_channels, expansion_factor=6):
        super(InvertedResidualBlock, self).__init__()
        
        hidden_dim = in_channels * expansion_factor
        self.use_residual = in_channels == out_channels
        
        layers = []
        
        # Expansion
        if expansion_factor != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise (linear)
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class AnimeResidualBlock(nn.Module):
    """애니메이션 고주파 복원을 위한 개선된 Residual Block"""
    def __init__(self, num_features):
        super(AnimeResidualBlock, self).__init__()
        
        # Multi-scale feature extraction
        self.multi_scale = MultiScaleConv(num_features, num_features)
        
        # Dilated convolution for context
        self.dilated_conv = DilatedConvBlock(num_features, num_features)
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(num_features)
        self.spatial_attention = SpatialAttention()
        
        # Final convolution
        self.final_conv = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        residual = x
        
        # Multi-scale feature extraction
        out = self.multi_scale(x)
        
        # Dilated convolution
        out = self.dilated_conv(out)
        
        # Channel attention
        out = self.channel_attention(out)
        
        # Spatial attention
        out = self.spatial_attention(out)
        
        # Final processing
        out = self.final_conv(out)
        out = self.bn(out)
        
        # Residual connection
        return out + residual

class EdgeEnhancementModule(nn.Module):
    """엣지 강화 모듈 - 애니메이션의 선명한 경계선 보존"""
    def __init__(self, in_channels):
        super(EdgeEnhancementModule, self).__init__()
        
        # Sobel edge detection kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        self.edge_conv = nn.Conv2d(2, in_channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Extract edges using Sobel filters
        if x.size(1) == 1:  # Grayscale
            edge_x = F.conv2d(x, self.sobel_x, padding=1)
            edge_y = F.conv2d(x, self.sobel_y, padding=1)
        else:  # Multi-channel
            edge_x = F.conv2d(x[:, 0:1], self.sobel_x, padding=1)
            edge_y = F.conv2d(x[:, 0:1], self.sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        edge_features = torch.cat([edge_x, edge_y], dim=1)
        
        # Generate edge attention map
        edge_attention = self.edge_conv(edge_features)
        edge_attention = self.sigmoid(edge_attention)
        
        return x * (1 + edge_attention)

class PixelShuffleUpsampling(nn.Module):
    """Sub-pixel convolution for efficient upsampling"""
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(PixelShuffleUpsampling, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels * (upscale_factor ** 2), 
            kernel_size=3, 
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x

class HighFrequencyGenerator(nn.Module):
    """저주파 입력에서 고주파 텍스처 생성에 특화된 Generator"""
    def __init__(self):
        super(HighFrequencyGenerator, self).__init__()
        
        num_features = config.gen_num_features
        num_residual_blocks = config.gen_num_residual_blocks
        
        # 저주파 특성 추출
        self.low_freq_encoder = nn.Sequential(
            nn.Conv2d(1, num_features, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )
        
        # 텍스처 컨텍스트 분석 (어떤 종류의 고주파가 필요한지 판단)
        self.context_analyzer = nn.Sequential(
            nn.Conv2d(num_features, num_features, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 고주파 텍스처 생성 블록들
        texture_blocks = []
        for _ in range(num_residual_blocks):
            texture_blocks.append(AnimeResidualBlock(num_features))
        self.texture_generator = nn.Sequential(*texture_blocks)
        
        # 고주파 residual 출력 (저주파에 더할 고주파 성분만 생성)
        self.hf_residual_output = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, 1, 5, padding=2),
            nn.Tanh()  # residual이므로 [-1, 1] 범위
        )
        
        # 적응적 가중치 (어느 정도 고주파를 더할지 결정)
        self.adaptive_weight = nn.Sequential(
            nn.Conv2d(num_features, num_features // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 4, 1, 1),
            nn.Sigmoid()  # [0, 1] 범위로 가중치 조절
        )
    
    def forward(self, x):
        # 저주파 입력 (이미 손상된 이미지)
        low_freq_features = self.low_freq_encoder(x)
        
        # 컨텍스트 분석 (어떤 텍스처가 필요한지)
        context = self.context_analyzer(low_freq_features)
        
        # 고주파 텍스처 생성
        hf_features = self.texture_generator(context)
        
        # 고주파 residual 생성
        hf_residual = self.hf_residual_output(hf_features)
        
        # 적응적 가중치 계산
        alpha = self.adaptive_weight(hf_features)
        
        # 최종 출력 = 원본 저주파 + 가중된 고주파 residual
        output = x + alpha * hf_residual
        
        return output

class HighFrequencyDiscriminator(nn.Module):
    """고주파 텍스처 품질 판별에 특화된 Discriminator"""
    def __init__(self):
        super(HighFrequencyDiscriminator, self).__init__()
        
        num_features = config.disc_num_features
        
        # 고주파 성분 추출 (입력에서 저주파 제거)
        self.hf_extractor = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Conv2d(1, 1, 3, padding=1),
        )
        
        # 텍스처 품질 분석
        layers = []
        
        # 첫 번째 레이어 - 고주파 텍스처 감지
        layers.extend([
            nn.Conv2d(1, num_features, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # 텍스처 패턴 분석 레이어들
        in_features = num_features
        for i in range(config.disc_num_layers - 1):
            out_features = min(num_features * (2 ** (i + 1)), 512)
            stride = 2 if i % 2 == 0 else 1
            
            # 텍스처 패턴에 집중하는 Multi-scale convolution
            layers.extend([
                MultiScaleConv(in_features, out_features),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            if stride == 2:
                layers.append(nn.AvgPool2d(2, 2))
            
            in_features = out_features
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 텍스처 품질 판별
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 고주파 성분에 집중 (저주파는 이미 손상되어 있으므로)
        hf_component = x - F.avg_pool2d(x, 3, stride=1, padding=1)  # 간단한 고주파 추출
        
        # 텍스처 품질 분석
        features = self.conv_layers(hf_component)
        
        # 최종 판별
        quality_score = self.classifier(features)
        return quality_score

# 모델 팩토리 함수
def create_anime_srgan_models():
    """애니메이션 고주파 텍스처 생성 최적화 모델들을 생성하는 팩토리 함수"""
    generator = HighFrequencyGenerator()
    discriminator = HighFrequencyDiscriminator()
    
    return generator, discriminator

# 하위 호환성을 위한 별칭
def create_fast_srgan_models():
    """Fast-SRGAN 모델들을 생성하는 팩토리 함수 (하위 호환성)"""
    return create_anime_srgan_models()

def count_parameters(model):
    """모델의 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 고주파 텍스처 생성 모델 테스트
    print("=" * 60)
    print("고주파 텍스처 생성 GAN 모델 테스트")
    print("=" * 60)
    
    generator, discriminator = create_anime_srgan_models()
    
    # 파라미터 수 출력
    gen_params = count_parameters(generator)
    disc_params = count_parameters(discriminator)
    
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    print(f"Total parameters: {gen_params + disc_params:,}")
    print()
    
    # 테스트: 저주파 손상된 이미지 → 고주파 텍스처 생성
    test_resolutions = [
        (128, 128),   # 작은 패치
        (256, 256),   # 중간 패치
        (576, 1024),  # 실제 사용 해상도
    ]
    
    for h, w in test_resolutions:
        # 저주파만 남은 손상된 이미지 시뮬레이션
        low_freq_input = torch.randn(1, 1, h, w) * 0.3  # 저주파 특성
        
        print(f"테스트 해상도: {h}x{w}")
        
        # Generator 테스트: 저주파 → 고주파 텍스처 추가
        with torch.no_grad():
            enhanced_output = generator(low_freq_input)
            
            # 생성된 고주파 residual 계산
            hf_residual = enhanced_output - low_freq_input
            
            print(f"  입력 (저주파): {low_freq_input.shape}")
            print(f"  출력 (향상됨): {enhanced_output.shape}")
            print(f"  고주파 residual 강도: {hf_residual.abs().mean().item():.4f}")
            
            # Discriminator 테스트: 텍스처 품질 판별
            quality_score = discriminator(enhanced_output)
            print(f"  텍스처 품질 점수: {quality_score.item():.4f}")
        print()
    
    print("고주파 텍스처 생성 GAN 모델 완성!")
    print()
    print("핵심 개념:")
    print("- 입력: 주파수 영역에서 고주파가 제거된 저주파 이미지")
    print("- 목표: 사라진 고주파를 '복원'이 아닌 '그럴듯하게 생성'")
    print("- 방법: 저주파 패턴 분석 → 적절한 고주파 텍스처 hallucination")
    print("- 출력: 입력 + 생성된 고주파 residual")
    print()
    print("주요 특징:")
    print("- Context Analyzer: 어떤 종류의 텍스처가 필요한지 분석")
    print("- Texture Generator: 애니메이션 특화 고주파 패턴 생성")
    print("- Adaptive Weighting: 영역별로 고주파 강도 조절")
    print("- HF-focused Discriminator: 고주파 텍스처 품질에만 집중")