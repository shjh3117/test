import os
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from config import ReconConfig, TrainConfig


class HighPassFilter(nn.Module):
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


class LowFreqDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        self.samples = []
        scene_dirs = sorted([d for d in glob(os.path.join(root_dir, '*')) if os.path.isdir(d)])
        for scene_dir in scene_dirs:
            x_dir = os.path.join(scene_dir, 'x')
            if not os.path.isdir(x_dir):
                continue
            for x_path in sorted(glob(os.path.join(x_dir, '*.png'))):
                frame_name = os.path.basename(x_path)
                self.samples.append((x_path, scene_dir, frame_name))

        if not self.samples:
            raise RuntimeError(f"No low-frequency frames found under {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x_path, scene_dir, frame_name = self.samples[idx]
        x_img = Image.open(x_path).convert('L')
        x_np = np.array(x_img, dtype=np.float32) / 255.0
        x_tensor = torch.from_numpy(x_np).unsqueeze(0)
        return {
            'x': x_tensor,
            'x_path': x_path,
            'scene_dir': scene_dir,
            'frame_name': frame_name,
        }


def load_generator(cfg_train: TrainConfig, device) -> Generator:
    checkpoint_path = ReconConfig.CHECKPOINT_PATH
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    generator = Generator(cfg_train.BASE_CHANNELS, cfg_train.NUM_RESIDUAL_BLOCKS, cfg_train.SCALE_FACTOR)
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()
    return generator


def reconstruct(cfg_recon: ReconConfig, cfg_train: TrainConfig) -> None:
    device = torch.device(cfg_recon.DEVICE)
    dataset = LowFreqDataset(cfg_recon.DATA_ROOT)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg_recon.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg_recon.NUM_WORKERS,
        pin_memory=cfg_recon.PIN_MEMORY,
    )

    generator = load_generator(cfg_train, device)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Reconstructing', ncols=110):
            x = batch['x'].to(device, non_blocking=True)
            preds = generator(x)

            preds_np = preds.cpu().numpy()
            for idx in range(preds_np.shape[0]):
                frame_name = batch['frame_name'][idx]
                scene_dir = batch['scene_dir'][idx]
                target_dir = os.path.join(scene_dir, cfg_recon.OUTPUT_SUBDIR)
                os.makedirs(target_dir, exist_ok=True)

                out_arr = np.clip(preds_np[idx, 0] * 255.0, 0, 255).astype(np.uint8)
                out_img = Image.fromarray(out_arr, mode='L')
                out_img.save(os.path.join(target_dir, frame_name), compress_level=1)


if __name__ == '__main__':
    reconstruct(ReconConfig, TrainConfig)
