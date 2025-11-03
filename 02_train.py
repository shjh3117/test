import os
import random
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from config import TrainConfig


# ===========================
# Utility Functions
# ===========================

def create_grad_scaler(enabled: bool):
    return torch.amp.GradScaler('cuda', enabled=enabled)

def autocast_ctx(enabled: bool):
    return torch.amp.autocast('cuda', enabled=enabled)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================
# Dataset
# ===========================

class SceneFrameDataset(Dataset):
    """Scene 디렉터리에서 x/Y 프레임 페어를 구성하는 Dataset."""

    def __init__(self, root_dir: str, augment: bool = False, samples=None) -> None:
        self.augment = augment
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._collect_samples(root_dir)

        if not self.samples:
            raise RuntimeError(f"No paired frames found under {root_dir}")

    @staticmethod
    def _collect_samples(root_dir: str):
        samples = []
        scene_dirs = sorted([d for d in glob(os.path.join(root_dir, '*')) if os.path.isdir(d)])
        for scene_dir in scene_dirs:
            x_dir = os.path.join(scene_dir, 'x')
            y_dir = os.path.join(scene_dir, 'Y')
            if not os.path.isdir(x_dir) or not os.path.isdir(y_dir):
                continue

            for x_path in sorted(glob(os.path.join(x_dir, '*.png'))):
                frame_name = os.path.basename(x_path)
                y_path = os.path.join(y_dir, frame_name)
                if os.path.isfile(y_path):
                    samples.append((x_path, y_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x_path, y_path = self.samples[idx]

        x_img = Image.open(x_path).convert('L')
        y_img = Image.open(y_path).convert('L')

        if self.augment and random.random() < 0.5:
            x_img = x_img.transpose(Image.FLIP_LEFT_RIGHT)
            y_img = y_img.transpose(Image.FLIP_LEFT_RIGHT)

        x_np = np.array(x_img, dtype=np.float32) / 255.0
        y_np = np.array(y_img, dtype=np.float32) / 255.0

        x_tensor = torch.from_numpy(x_np).unsqueeze(0)
        y_tensor = torch.from_numpy(y_np).unsqueeze(0)

        return {
            'x': x_tensor,
            'y': y_tensor,
            'x_path': x_path,
            'y_path': y_path,
        }


# ===========================
# Network Modules
# ===========================

class ResidualBlock(nn.Module):
    """Residual Block with PReLU activation"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class UpsampleBlock(nn.Module):
    """PixelShuffle 기반 2x 업샘플링 블록"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class LowPassFilter(nn.Module):
    """Gaussian 기반 저주파 통과 필터"""
    
    def __init__(self, kernel_size: int = 21, sigma: float = 5.0) -> None:
        super().__init__()
        # 큰 커널로 더 넓은 저주파 범위 커버
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        
        self.register_buffer('kernel', kernel.view(1, 1, kernel_size, kernel_size))
        self.padding = kernel_size // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, padding=self.padding)


class EdgeDetector(nn.Module):
    """Sobel 기반 에지 검출"""
    
    def __init__(self) -> None:
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]])
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        return edge


# ===========================
# Generator (저주파 보존 + 고주파 생성)
# ===========================

class Generator(nn.Module):
    """
    저주파 이미지를 입력으로 받아 고해상도 이미지 생성
    핵심: 입력 저주파를 직접 보존하고, 고주파만 생성하여 더함
    """
    def __init__(self, base_channels: int = 64, num_residual_blocks: int = 16, scale_factor: int = 5) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        
        # 1. 입력 처리
        self.entry = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        # 2. 깊은 Residual Blocks (특징 추출)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_residual_blocks)]
        )
        
        # 3. 중간 압축
        self.mid_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels)
        )
        
        # 4. 업샘플링 (256x144 -> 512x288 -> 1024x576 -> 1280x720)
        # scale_factor=5 이므로 먼저 2배씩 업샘플링
        self.upsample_blocks = nn.ModuleList([
            UpsampleBlock(base_channels, base_channels),  # 2x
            UpsampleBlock(base_channels, base_channels),  # 2x
        ])
        
        # 5. 최종 조정 (4배 업샘플링 후 1.25배 보간으로 5배 달성)
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels, 1, kernel_size=9, padding=4)
        )
        
        # 6. 저주파 필터 (원본 보존 확인용)
        self.lowpass_filter = LowPassFilter(kernel_size=21, sigma=5.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 144, 256] 저주파 이미지
        Returns:
            [B, 1, 720, 1280] 고해상도 복원 이미지
        """
        batch_size = x.size(0)
        
        # 저주파 원본을 bilinear로 업스케일 (저주파 보존 base)
        x_upscaled = F.interpolate(
            x, 
            size=(720, 1280),  # 직접 5배 업스케일
            mode='bilinear', 
            align_corners=False
        )
        
        # 저주파 성분 추출 (더 명확한 저주파 보존을 위해)
        x_lowfreq_base = self.lowpass_filter(x_upscaled)
        
        # 특징 추출 및 고주파 생성 경로
        feat = self.entry(x_upscaled)
        feat_res = self.res_blocks(feat)
        feat_res = self.mid_conv(feat_res) + feat  # 전역 residual
        
        # 점진적 업샘플링 (이미 720x1280이므로 특징 정제에 사용)
        # 실제로는 x_upscaled가 이미 목표 크기이므로 특징만 정제
        highfreq_feat = feat_res
        
        # 고주파 성분 생성
        highfreq_output = self.final_conv(highfreq_feat)
        
        # 핵심: 저주파 base + 생성된 고주파 = 최종 출력
        # 이렇게 하면 저주파는 원본이 직접 보존됨
        output = x_lowfreq_base + highfreq_output
        
        return output.clamp(0.0, 1.0)


# ===========================
# Discriminator (PatchGAN, 멀티스케일)
# ===========================

class PatchDiscriminator(nn.Module):
    """PatchGAN 스타일 판별기 (70x70 패치)"""
    
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        
        # 입력: [B, 2, H, W] (x와 y를 concat)
        layers = [
            nn.Conv2d(2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 깊은 conv 레이어
        channel_sizes = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        for i in range(len(channel_sizes) - 1):
            in_c = channel_sizes[i]
            out_c = channel_sizes[i + 1]
            stride = 2 if i < 2 else 1
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # 최종 판별 레이어
        layers += [nn.Conv2d(channel_sizes[-1], 1, kernel_size=4, padding=1)]
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 저주파 이미지 (업스케일된)
            y: 타겟/생성 이미지
        Returns:
            패치별 판별 결과
        """
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)


class MultiScaleDiscriminator(nn.Module):
    """3개의 스케일에서 판별 (원본, 1/2, 1/4)"""
    
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(base_channels) for _ in range(3)
        ])
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> list:
        results = []
        x_scaled, y_scaled = x, y
        
        for i, disc in enumerate(self.discriminators):
            results.append(disc(x_scaled, y_scaled))
            if i < 2:  # 마지막 스케일 제외
                x_scaled = self.downsample(x_scaled)
                y_scaled = self.downsample(y_scaled)
        
        return results


# ===========================
# Loss Functions
# ===========================

def compute_lowfreq_preservation_loss(pred: torch.Tensor, target: torch.Tensor, 
                                       lowpass_filter: LowPassFilter) -> torch.Tensor:
    """저주파 보존 손실 (가장 중요!)"""
    pred_lowfreq = lowpass_filter(pred)
    target_lowfreq = lowpass_filter(target)
    return F.l1_loss(pred_lowfreq, target_lowfreq)


def compute_edge_loss(pred: torch.Tensor, target: torch.Tensor, 
                      edge_detector: EdgeDetector) -> torch.Tensor:
    """에지 보존 손실 (애니메이션 윤곽선)"""
    pred_edge = edge_detector(pred)
    target_edge = edge_detector(target)
    return F.l1_loss(pred_edge, target_edge)


def compute_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """그래디언트 손실 (샤프니스)"""
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    loss_dx = F.l1_loss(pred_dx, target_dx)
    loss_dy = F.l1_loss(pred_dy, target_dy)
    return loss_dx + loss_dy


# ===========================
# Training Functions
# ===========================

def prepare_dataloaders(cfg: TrainConfig):
    full_dataset = SceneFrameDataset(cfg.DATA_ROOT, augment=False)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    val_size = max(1, int(len(full_dataset) * 0.1))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]

    train_dataset = SceneFrameDataset(cfg.DATA_ROOT, augment=True, samples=train_samples)
    val_dataset = SceneFrameDataset(cfg.DATA_ROOT, augment=False, samples=val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=len(train_dataset) >= cfg.BATCH_SIZE,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=max(1, cfg.NUM_WORKERS // 2),
        pin_memory=cfg.PIN_MEMORY,
    )

    return train_loader, val_loader


def save_checkpoint(generator, discriminator, optim_g, optim_d, epoch, cfg: TrainConfig, val_metrics: dict) -> None:
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_g': optim_g.state_dict(),
        'optimizer_d': optim_d.state_dict(),
        'val_metrics': val_metrics,
    }
    torch.save(checkpoint, os.path.join(cfg.MODEL_DIR, f'epoch_{epoch:04d}_checkpoint.pt'))
    torch.save(generator.state_dict(), os.path.join(cfg.MODEL_DIR, 'latest_generator.pt'))


def validate(generator, data_loader, device, cfg: TrainConfig, 
             lowpass_filter: LowPassFilter, edge_detector: EdgeDetector) -> dict:
    """검증 수행"""
    generator.eval()
    total_l1 = 0.0
    total_lowfreq = 0.0
    total_edge = 0.0
    total_psnr = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)

            preds = generator(x)
            
            # L1 손실
            l1 = F.l1_loss(preds, y, reduction='sum')
            total_l1 += l1.item()
            
            # 저주파 보존 손실
            lowfreq_loss = compute_lowfreq_preservation_loss(preds, y, lowpass_filter)
            total_lowfreq += lowfreq_loss.item() * y.size(0)
            
            # 에지 손실
            edge_loss = compute_edge_loss(preds, y, edge_detector)
            total_edge += edge_loss.item() * y.size(0)
            
            # PSNR
            mse = F.mse_loss(preds, y, reduction='none').mean(dim=[1, 2, 3])
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            total_psnr += psnr.sum().item()
            
            total_samples += y.size(0)

    generator.train()
    
    num_pixels = total_samples * y.size(2) * y.size(3)
    
    return {
        'l1': total_l1 / max(1, num_pixels),
        'lowfreq': total_lowfreq / max(1, total_samples),
        'edge': total_edge / max(1, total_samples),
        'psnr': total_psnr / max(1, total_samples),
    }


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    print("="*60)
    print("Training: Low-Frequency Preserving SR-GAN")
    print("="*60)

    # 데이터 로더
    train_loader, val_loader = prepare_dataloaders(cfg)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # 모델 생성
    generator = Generator(
        base_channels=cfg.BASE_CHANNELS, 
        num_residual_blocks=cfg.NUM_RESIDUAL_BLOCKS, 
        scale_factor=cfg.SCALE_FACTOR
    ).to(device)
    
    discriminator = MultiScaleDiscriminator(base_channels=cfg.BASE_CHANNELS).to(device)
    
    # 필터 생성
    lowpass_filter = LowPassFilter(kernel_size=21, sigma=5.0).to(device)
    edge_detector = EdgeDetector().to(device)
    
    # 옵티마이저
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=cfg.LR_GENERATOR, betas=(cfg.BETA1, cfg.BETA2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.LR_DISCRIMINATOR, betas=(cfg.BETA1, cfg.BETA2))

    # AMP 설정
    use_amp = device.type == 'cuda'
    scaler_g = create_grad_scaler(use_amp)
    scaler_d = create_grad_scaler(use_amp)
    
    # GAN 손실
    criterion_gan = nn.BCEWithLogitsLoss()

    global_step = 0
    
    print(f"Device: {device}")
    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    print("="*60)

    for epoch in range(1, cfg.EPOCHS + 1):
        generator.train()
        discriminator.train()
        
        progress = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.EPOCHS}', ncols=120)
        
        for batch in progress:
            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            
            # x를 업스케일 (판별기 입력용)
            x_upscaled = F.interpolate(x, size=(720, 1280), mode='bilinear', align_corners=False)

            # ===========================
            # Discriminator 업데이트
            # ===========================
            optimizer_d.zero_grad(set_to_none=True)
            
            with autocast_ctx(use_amp):
                fake_images = generator(x).detach()
                
                # 멀티스케일 판별
                real_logits_list = discriminator(x_upscaled, y)
                fake_logits_list = discriminator(x_upscaled, fake_images)
                
                loss_d_real = 0
                loss_d_fake = 0
                for real_logits, fake_logits in zip(real_logits_list, fake_logits_list):
                    loss_d_real += criterion_gan(real_logits, torch.ones_like(real_logits))
                    loss_d_fake += criterion_gan(fake_logits, torch.zeros_like(fake_logits))
                
                loss_d = 0.5 * (loss_d_real + loss_d_fake) / len(real_logits_list)
            
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            # ===========================
            # Generator 업데이트
            # ===========================
            optimizer_g.zero_grad(set_to_none=True)
            
            with autocast_ctx(use_amp):
                preds = generator(x)
                
                # 1. Adversarial Loss (멀티스케일)
                pred_logits_list = discriminator(x_upscaled, preds)
                adv_loss = 0
                for pred_logits in pred_logits_list:
                    adv_loss += criterion_gan(pred_logits, torch.ones_like(pred_logits))
                adv_loss = adv_loss / len(pred_logits_list)
                
                # 2. Pixel Loss (L1)
                pixel_loss = F.l1_loss(preds, y)
                
                # 3. 저주파 보존 손실 (핵심!)
                lowfreq_loss = compute_lowfreq_preservation_loss(preds, y, lowpass_filter)
                
                # 4. 에지 보존 손실 (애니메이션 윤곽선)
                edge_loss = compute_edge_loss(preds, y, edge_detector)
                
                # 5. 그래디언트 손실 (샤프니스)
                grad_loss = compute_gradient_loss(preds, y)
                
                # 총 Generator 손실
                loss_g = (
                    adv_loss * 1.0 +
                    pixel_loss * cfg.LAMBDA_L1 +
                    lowfreq_loss * cfg.LAMBDA_FREQ +
                    edge_loss * 10.0 +
                    grad_loss * 5.0
                )
            
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            global_step += 1

            if global_step % cfg.LOG_INTERVAL == 0:
                progress.set_postfix({
                    'G': f'{loss_g.item():.2f}',
                    'D': f'{loss_d.item():.3f}',
                    'L1': f'{pixel_loss.item():.4f}',
                    'LF': f'{lowfreq_loss.item():.4f}',
                    'E': f'{edge_loss.item():.4f}',
                })

        # Validation
        val_metrics = validate(generator, val_loader, device, cfg, lowpass_filter, edge_detector)

        # Checkpoint 저장
        if epoch % cfg.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, cfg, val_metrics)

        print(f'\n[Epoch {epoch}] Val - L1: {val_metrics["l1"]:.6f} | '
              f'LowFreq: {val_metrics["lowfreq"]:.6f} | '
              f'Edge: {val_metrics["edge"]:.6f} | '
              f'PSNR: {val_metrics["psnr"]:.2f}dB\n')
    
    print("="*60)
    print("Training completed!")
    print("="*60)


if __name__ == '__main__':
    train(TrainConfig)

