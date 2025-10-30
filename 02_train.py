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


# Use torch.amp API (PyTorch 1.9+)
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


class ResidualBlock(nn.Module):
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


class HighPassFilter(nn.Module):
    """고역 통과 필터로 저주파 성분을 보호."""

    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[-1.0, -1.0, -1.0],
             [-1.0, 8.0, -1.0],
             [-1.0, -1.0, -1.0]],
            dtype=torch.float32,
        )
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, padding=1)


class Generator(nn.Module):
    def __init__(self, base_channels: int, num_residual_blocks: int, scale_factor: int) -> None:
        super().__init__()
        self.scale_factor = scale_factor

        self.entry = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_residual_blocks)]
        )

        self.mid_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
        )

        self.exit = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.high_pass = HighPassFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False,
        )

        feat = self.entry(base)
        residual = self.res_blocks(feat)
        residual = self.mid_conv(residual) + feat
        residual = self.exit(residual)
        high_freq = self.high_pass(residual)
        out = base + high_freq
        return out.clamp(0.0, 1.0)


class Discriminator(nn.Module):
    def __init__(self, base_channels: int) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channel_sizes = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        for idx in range(len(channel_sizes) - 1):
            in_c = channel_sizes[idx]
            out_c = channel_sizes[idx + 1]
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers += [
            nn.Conv2d(channel_sizes[-1], channel_sizes[-1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_sizes[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_sizes[-1], 1, kernel_size=3, stride=1, padding=1),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def save_checkpoint(generator, discriminator, optim_g, optim_d, epoch, cfg: TrainConfig, val_l1: float) -> None:
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_g': optim_g.state_dict(),
        'optimizer_d': optim_d.state_dict(),
        'val_l1': val_l1,
    }
    torch.save(checkpoint, os.path.join(cfg.MODEL_DIR, f'epoch_{epoch:04d}_checkpoint.pt'))
    torch.save(generator.state_dict(), os.path.join(cfg.MODEL_DIR, 'latest_generator.pt'))


def validate(generator, data_loader, device, cfg: TrainConfig) -> float:
    generator.eval()
    total_l1 = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)

            preds = generator(x)
            l1 = F.l1_loss(preds, y, reduction='sum')

            total_l1 += l1.item()
            total_samples += y.size(0) * y.size(2) * y.size(3)

    generator.train()
    return total_l1 / max(1, total_samples)


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    train_loader, val_loader = prepare_dataloaders(cfg)

    generator = Generator(cfg.BASE_CHANNELS, cfg.NUM_RESIDUAL_BLOCKS, cfg.SCALE_FACTOR).to(device)
    discriminator = Discriminator(cfg.BASE_CHANNELS).to(device)

    criterion_gan = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=cfg.LR_GENERATOR, betas=(cfg.BETA1, cfg.BETA2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.LR_DISCRIMINATOR, betas=(cfg.BETA1, cfg.BETA2))

    use_amp = device.type == 'cuda'
    scaler_g = create_grad_scaler(use_amp)
    scaler_d = create_grad_scaler(use_amp)

    global_step = 0

    for epoch in range(1, cfg.EPOCHS + 1):
        progress = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.EPOCHS}', ncols=110)
        for batch in progress:
            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)

            # Discriminator update
            optimizer_d.zero_grad(set_to_none=True)
            with autocast_ctx(use_amp):
                fake_images = generator(x).detach()
                real_logits = discriminator(y)
                fake_logits = discriminator(fake_images)
                loss_d_real = criterion_gan(real_logits, torch.ones_like(real_logits))
                loss_d_fake = criterion_gan(fake_logits, torch.zeros_like(fake_logits))
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            # Generator update
            optimizer_g.zero_grad(set_to_none=True)
            with autocast_ctx(use_amp):
                preds = generator(x)
                pred_logits = discriminator(preds)
                adv_loss = criterion_gan(pred_logits, torch.ones_like(pred_logits))
                l1_loss = F.l1_loss(preds, y) * cfg.LAMBDA_L1
                loss_g = adv_loss + l1_loss
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            global_step += 1

            if global_step % cfg.LOG_INTERVAL == 0:
                progress.set_postfix({
                    'loss_g': f'{loss_g.item():.3f}',
                    'loss_d': f'{loss_d.item():.3f}',
                    'l1': f'{l1_loss.item():.3f}',
                })

        val_l1 = validate(generator, val_loader, device, cfg)

        if epoch % cfg.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, cfg, val_l1)

        print(f'Epoch {epoch}: val L1 = {val_l1:.6f}')


if __name__ == '__main__':
    train(TrainConfig)
