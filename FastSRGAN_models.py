"""
Fast-SRGAN (MobileSRGAN) 모델 정의
모바일 및 실시간 처리에 최적화된 경량 SRGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from FastSRGANconfig import fast_srgan_config as config

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

class MobileResidualBlock(nn.Module):
    """Fast-SRGAN용 경량화된 Residual Block"""
    def __init__(self, num_features):
        super(MobileResidualBlock, self).__init__()
        
        if config.use_mobile_blocks:
            self.conv1 = DepthwiseSeparableConv(num_features, num_features)
            self.conv2 = DepthwiseSeparableConv(num_features, num_features)
        else:
            self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(num_features)
        self.bn2 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        if config.use_mobile_blocks:
            out = self.conv1(x)
            out = self.conv2(out)
        else:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        
        return out + residual

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

class FastSRGANGenerator(nn.Module):
    """Fast-SRGAN Generator (MobileNet 기반)"""
    def __init__(self):
        super(FastSRGANGenerator, self).__init__()
        
        num_features = config.gen_num_features
        num_residual_blocks = config.gen_num_residual_blocks
        upscale_factor = config.gen_upscale_factor
        
        # 첫 번째 레이어
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_features, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        residual_blocks = []
        for _ in range(num_residual_blocks):
            residual_blocks.append(MobileResidualBlock(num_features))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # Post-residual conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features)
        )
        
        # Upsampling layers
        upsampling_layers = []
        for _ in range(int(torch.log2(torch.tensor(upscale_factor)).item())):
            upsampling_layers.append(
                PixelShuffleUpsampling(num_features, num_features, 2)
            )
        self.upsampling = nn.Sequential(*upsampling_layers)
        
        # 마지막 레이어
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_features, 1, kernel_size=9, padding=4),
            nn.Tanh()
        )
    
    def forward(self, x):
        # 첫 번째 레이어
        out1 = self.conv1(x)
        
        # Residual blocks
        out = self.residual_blocks(out1)
        
        # Post-residual conv + skip connection
        out2 = self.conv2(out)
        out = out1 + out2
        
        # Upsampling
        out = self.upsampling(out)
        
        # 마지막 레이어
        out = self.conv3(out)
        
        return out

class FastSRGANDiscriminator(nn.Module):
    """Fast-SRGAN Discriminator (경량화)"""
    def __init__(self):
        super(FastSRGANDiscriminator, self).__init__()
        
        num_features = config.disc_num_features
        
        layers = []
        
        # 첫 번째 레이어
        layers.extend([
            nn.Conv2d(1, num_features, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # 중간 레이어들
        in_features = num_features
        for i in range(config.disc_num_layers - 1):
            out_features = min(num_features * (2 ** (i + 1)), 512)
            stride = 2 if i % 2 == 0 else 1
            
            if config.use_mobile_blocks and i > 0:
                layers.append(DepthwiseSeparableConv(
                    in_features, out_features, stride=stride, padding=1
                ))
            else:
                layers.extend([
                    nn.Conv2d(in_features, out_features, kernel_size=3, 
                             stride=stride, padding=1),
                    nn.BatchNorm2d(out_features),
                    nn.LeakyReLU(0.2, inplace=True)
                ])
            
            in_features = out_features
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 분류 레이어
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# 모델 팩토리 함수
def create_fast_srgan_models():
    """Fast-SRGAN 모델들을 생성하는 팩토리 함수"""
    generator = FastSRGANGenerator()
    discriminator = FastSRGANDiscriminator()
    
    return generator, discriminator

def count_parameters(model):
    """모델의 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 모델 테스트
    generator, discriminator = create_fast_srgan_models()
    
    # 파라미터 수 출력
    gen_params = count_parameters(generator)
    disc_params = count_parameters(discriminator)
    
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    print(f"Total parameters: {gen_params + disc_params:,}")
    
    # 더미 입력으로 테스트
    dummy_input = torch.randn(1, 1, 128, 128)
    
    # Generator 테스트
    with torch.no_grad():
        gen_output = generator(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Generator output shape: {gen_output.shape}")
        
        # Discriminator 테스트
        disc_output = discriminator(gen_output)
        print(f"Discriminator output shape: {disc_output.shape}")
        print(f"Discriminator output: {disc_output.item():.4f}")
    
    print("Fast-SRGAN models created successfully!")