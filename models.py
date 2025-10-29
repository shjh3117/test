"""
DeblurGAN-v2: 주파수 도메인 DC 투영 기반 고주파 복원 모델
FPN 백본 + Relativistic GAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


# ============================================================================
# FPN Backbone (DeblurGAN-v2)
# ============================================================================

class ConvBlock(nn.Module):
    """기본 Convolution Block"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual Block"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class FPNEncoder(nn.Module):
    """FPN 인코더 - 다중 스케일 특성 추출"""
    def __init__(self, in_channels=1):
        super().__init__()
        
        # Downsampling path
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, 64, 7, 1, 3),
            ResBlock(64)
        )
        
        self.enc2 = nn.Sequential(
            ConvBlock(64, 128, 3, 2, 1),
            ResBlock(128)
        )
        
        self.enc3 = nn.Sequential(
            ConvBlock(128, 256, 3, 2, 1),
            ResBlock(256)
        )
        
        self.enc4 = nn.Sequential(
            ConvBlock(256, 512, 3, 2, 1),
            ResBlock(512)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)  # 1x
        e2 = self.enc2(e1)  # 1/2x
        e3 = self.enc3(e2)  # 1/4x
        e4 = self.enc4(e3)  # 1/8x
        return e1, e2, e3, e4


class FPNDecoder(nn.Module):
    """FPN 디코더 - 다중 스케일 특성 융합"""
    def __init__(self, out_channels=1):
        super().__init__()
        
        # Lateral connections
        self.lat4 = nn.Conv2d(512, 256, 1)
        self.lat3 = nn.Conv2d(256, 256, 1)
        self.lat2 = nn.Conv2d(128, 256, 1)
        self.lat1 = nn.Conv2d(64, 256, 1)
        
        # Upsampling
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Smoothing
        self.smooth4 = ResBlock(256)
        self.smooth3 = ResBlock(256)
        self.smooth2 = ResBlock(256)
        self.smooth1 = ResBlock(256)
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, 1, 1)
        )
    
    def forward(self, features):
        e1, e2, e3, e4 = features
        
        # Top-down pathway
        p4 = self.lat4(e4)
        p4 = self.smooth4(p4)
        
        p3 = self.lat3(e3) + self.up4(p4)
        p3 = self.smooth3(p3)
        
        p2 = self.lat2(e2) + self.up3(p3)
        p2 = self.smooth2(p2)
        
        p1 = self.lat1(e1) + self.up2(p2)
        p1 = self.smooth1(p1)
        
        # Final output
        out = self.final(p1)
        return out


class Generator(nn.Module):
    """
    DeblurGAN-v2 Generator with DC Projection
    입력: x (저주파만 있는 이미지)
    출력: 고주파 residual → x + residual → DC 투영
    """
    def __init__(self):
        super().__init__()
        
        self.encoder = FPNEncoder(in_channels=1)
        self.decoder = FPNDecoder(out_channels=1)
    
    def forward(self, x):
        """
        x: 저주파 입력 [B, 1, H, W]
        return: 고주파 residual [B, 1, H, W]
        """
        features = self.encoder(x)
        hf_residual = self.decoder(features)
        return hf_residual
    
    def forward_with_dc(self, x, mask=None):
        """
        DC 투영 포함 forward
        x: 저주파 입력
        mask: 주파수 마스크
        return: DC 투영된 최종 출력
        """
        # 고주파 residual 예측
        hf_residual = self.forward(x)
        
        # 임시 복원
        y_tilde = x + hf_residual
        
        # DC 투영
        if mask is not None and config.use_hard_dc:
            y_hat = dc_projection(x, y_tilde, mask)
            return y_hat
        else:
            return y_tilde


# ============================================================================
# Discriminator (Relativistic PatchGAN)
# ============================================================================

class Discriminator(nn.Module):
    """
    Relativistic PatchGAN Discriminator
    고주파 텍스처/에지 품질 판별
    """
    def __init__(self):
        super().__init__()
        
        # PatchGAN layers
        self.model = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 31x31
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 31x31 -> 30x30
            nn.Conv2d(512, 1, 4, 1, 1)
        )
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# 유틸리티 함수
# ============================================================================

def dc_projection(x_low, y_tilde, mask):
    """
    하드 DC 투영: F(ŷ) = M ⊙ F(x) + (1−M) ⊙ F(ỹ)
    x_low: 저주파 입력 [B, 1, H, W]
    y_tilde: 임시 복원 [B, 1, H, W]
    mask: 주파수 마스크 [1, 1, H, W] (1=저주파, 0=고주파)
    """
    # FFT
    x_fft = torch.fft.fft2(x_low.squeeze(1), dim=(-2, -1))
    y_fft = torch.fft.fft2(y_tilde.squeeze(1), dim=(-2, -1))
    mask_2d = mask.squeeze(1)
    
    # DC 투영
    y_hat_fft = mask_2d * x_fft + (1 - mask_2d) * y_fft
    
    # IFFT
    y_hat = torch.fft.ifft2(y_hat_fft, dim=(-2, -1)).real
    
    return y_hat.unsqueeze(1)


def create_frequency_mask(size, center_ratio=0.1, device='cpu'):
    """
    주파수 마스크 생성
    center_ratio: 중앙 저주파 영역 비율
    return: mask [1, 1, H, W] (1=저주파, 0=고주파)
    """
    H, W = size
    mask = torch.zeros(1, 1, H, W, device=device)
    
    center_h = int(H * center_ratio / 2)
    center_w = int(W * center_ratio / 2)
    
    h_start = H // 2 - center_h
    h_end = H // 2 + center_h
    w_start = W // 2 - center_w
    w_end = W // 2 + center_w
    
    mask[:, :, h_start:h_end, w_start:w_end] = 1
    
    return mask


# ============================================================================
# 테스트
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DeblurGAN-v2 기반 고주파 복원 모델")
    print("=" * 70)
    
    # 모델 생성
    generator = Generator()
    discriminator = Discriminator()
    
    # 파라미터 수
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"Generator: {gen_params:,} parameters")
    print(f"Discriminator: {disc_params:,} parameters")
    print(f"Total: {gen_params + disc_params:,} parameters")
    print()
    
    # 테스트
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    x = torch.randn(2, 1, 256, 256).to(device)
    mask = create_frequency_mask((256, 256), center_ratio=0.1, device=device)
    
    print(f"Device: {device}")
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print()
    
    # Generator forward
    with torch.no_grad():
        hf_residual = generator(x)
        y_hat = generator.forward_with_dc(x, mask)
        
        print(f"HF Residual shape: {hf_residual.shape}")
        print(f"Output (with DC) shape: {y_hat.shape}")
        print(f"HF Residual range: [{hf_residual.min():.3f}, {hf_residual.max():.3f}]")
        print()
        
        # Discriminator forward
        disc_out = discriminator(y_hat)
        print(f"Discriminator output shape: {disc_out.shape}")
        print()
    
    print("=" * 70)
    print("모델 테스트 완료!")
    print("=" * 70)
