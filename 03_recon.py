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
    """
    저주파 이미지를 입력으로 받아 고해상도 이미지 생성
    핵심: 입력 x의 bilinear 업스케일을 base로 보존하고, 네트워크는 고주파 디테일만 생성
    """
    def __init__(self, base_channels: int = 64, num_residual_blocks: int = 16, scale_factor: int = 5) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        
        # 1. 입력 처리 (업스케일된 저주파 이미지 기반)
        self.entry = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        # 2. 깊은 Residual Blocks (고주파 특징 추출)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_residual_blocks)]
        )
        
        # 3. 중간 압축
        self.mid_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels)
        )
        
        # 4. 최종 고주파 생성 레이어
        self.highfreq_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels, 1, kernel_size=9, padding=4),
            nn.Tanh()  # 고주파는 [-1, 1] 범위의 디테일
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 144, 256] 저주파 이미지
        Returns:
            [B, 1, 720, 1280] 고해상도 복원 이미지
        """
        # 핵심: 입력 저주파를 bilinear로 업스케일 = base (저주파 보존!)
        lowfreq_base = F.interpolate(
            x, 
            size=(720, 1280) if self.scale_factor == 5 else None,
            scale_factor=None if self.scale_factor == 5 else self.scale_factor,
            mode='bilinear', 
            align_corners=False
        )
        
        # 네트워크는 고주파 디테일만 생성
        feat = self.entry(lowfreq_base)
        feat_res = self.res_blocks(feat)
        feat_res = self.mid_conv(feat_res) + feat
        
        # 고주파 성분만 생성
        highfreq_detail = self.highfreq_conv(feat_res)
        
        # 최종 출력 = 저주파 base + 고주파 detail
        output = lowfreq_base + highfreq_detail * 0.1
        
        return output.clamp(0.0, 1.0)


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
