"""
DeblurGAN-v2 훈련 스크립트
주파수 도메인 DC 투영 + Relativistic GAN
"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.models as models
import torch.nn.functional as F

from config import config
from models import Generator, Discriminator, create_frequency_mask, dc_projection


# ============================================================================
# 손실 함수
# ============================================================================

class PerceptualLoss(nn.Module):
    """VGG 기반 Perceptual Loss"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16]).eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred, target):
        # Grayscale to RGB
        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)
        
        # Normalize
        pred_norm = (pred_rgb - self.mean) / self.std
        target_norm = (target_rgb - self.mean) / self.std
        
        # Extract features
        pred_feat = self.feature_extractor(pred_norm)
        target_feat = self.feature_extractor(target_norm)
        
        return F.l1_loss(pred_feat, target_feat)


class CombinedLoss(nn.Module):
    """
    복합 손실 함수:
    - λ_pix: 픽셀 L1
    - λ_perc: Perceptual (VGG)
    - λ_fft: 주파수 대역 가중 L1
    - λ_dc: DC 일치 (soft)
    - λ_adv: Relativistic Adversarial
    """
    def __init__(self):
        super().__init__()
        self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred, target, mask, disc_pred_fake=None, disc_pred_real=None):
        losses = {}
        
        # 1. Pixel L1 loss
        loss_pix = F.l1_loss(pred, target)
        losses['pix'] = loss_pix
        
        # 2. Perceptual loss
        loss_perc = self.perceptual_loss(pred, target)
        losses['perc'] = loss_perc
        
        # 3. Frequency-weighted L1 loss
        pred_fft = torch.fft.fft2(pred.squeeze(1), dim=(-2, -1))
        target_fft = torch.fft.fft2(target.squeeze(1), dim=(-2, -1))
        mask_2d = mask.squeeze(1)
        
        # 고주파 대역만 비교 (1-M)
        hf_mask = 1 - mask_2d
        loss_fft = F.l1_loss(torch.abs(hf_mask * pred_fft), torch.abs(hf_mask * target_fft))
        losses['fft'] = loss_fft
        
        # 4. DC consistency (soft) - optional
        if config.lambda_dc > 0:
            loss_dc = F.mse_loss(torch.abs(mask_2d * pred_fft), torch.abs(mask_2d * target_fft))
            losses['dc'] = loss_dc
        else:
            losses['dc'] = torch.tensor(0.0, device=pred.device)
        
        # 5. Relativistic Adversarial loss
        if disc_pred_fake is not None and disc_pred_real is not None:
            # Relativistic average discriminator
            loss_adv = F.mse_loss(disc_pred_fake - disc_pred_real.mean(), 
                                   torch.ones_like(disc_pred_fake))
            losses['adv'] = loss_adv
        else:
            losses['adv'] = torch.tensor(0.0, device=pred.device)
        
        # Total loss
        total_loss = (
            config.lambda_pix * losses['pix'] +
            config.lambda_perc * losses['perc'] +
            config.lambda_fft * losses['fft'] +
            config.lambda_dc * losses['dc'] +
            config.lambda_adv * losses['adv']
        )
        
        losses['total'] = total_loss
        return total_loss, losses


# ============================================================================
# 데이터셋
# ============================================================================

class HFDataset(Dataset):
    """고주파 복원 데이터셋"""
    def __init__(self):
        self.data_pairs = []
        
        # work_dir의 모든 scene 폴더
        video_dirs = [d for d in os.listdir(config.work_dir) 
                      if os.path.isdir(os.path.join(config.work_dir, d))]
        
        for video_dir in video_dirs:
            target_dir = os.path.join(config.work_dir, video_dir, config.target_channel)
            input_dir = os.path.join(config.work_dir, video_dir, config.input_channel)
            
            if os.path.exists(target_dir) and os.path.exists(input_dir):
                target_files = sorted(glob.glob(os.path.join(target_dir, "*.png")))
                
                # 첫 프레임만 사용
                if len(target_files) > 0:
                    first_frame = target_files[0]
                    frame_name = os.path.basename(first_frame)
                    input_file = os.path.join(input_dir, frame_name)
                    if os.path.exists(input_file):
                        self.data_pairs.append((input_file, first_frame))
                        print(f"Added: {video_dir}/{frame_name}")
        
        print(f"Total training pairs: {len(self.data_pairs)}")
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        input_path, target_path = self.data_pairs[idx]
        
        # Load images
        input_img = Image.open(input_path).convert('L')
        target_img = Image.open(target_path).convert('L')
        
        # To tensor [0, 1]
        input_tensor = torch.from_numpy(np.array(input_img, dtype=np.float32) / 255.0).unsqueeze(0)
        target_tensor = torch.from_numpy(np.array(target_img, dtype=np.float32) / 255.0).unsqueeze(0)
        
        return input_tensor, target_tensor


# ============================================================================
# 훈련 함수
# ============================================================================

def train():
    """메인 훈련 루프"""
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # 모델 생성
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # 옵티마이저
    optimizer_G = optim.Adam(generator.parameters(), lr=config.learning_rate_gen, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.learning_rate_disc, betas=(0.5, 0.999))
    
    # 손실 함수
    criterion = CombinedLoss().to(device)
    
    # 데이터셋
    dataset = HFDataset()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, 
                           num_workers=config.num_workers, pin_memory=True)
    
    # 주파수 마스크 생성 (첫 배치로부터)
    first_input, _ = dataset[0]
    H, W = first_input.shape[1:]
    freq_mask = create_frequency_mask((H, W), center_ratio=0.1, device=device)
    print(f"Frequency mask: {freq_mask.shape}, H={H}, W={W}")
    
    best_loss = float('inf')
    
    # 훈련 루프
    for epoch in range(config.num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_losses = {
            'G': 0.0, 'pix': 0.0, 'perc': 0.0, 'fft': 0.0, 'adv': 0.0, 'D': 0.0
        }
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, (input_img, target_img) in enumerate(pbar):
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            
            # ================== Train Generator ==================
            optimizer_G.zero_grad()
            
            # Forward
            hf_residual = generator(input_img)
            y_tilde = input_img + hf_residual
            
            # DC projection
            if config.use_hard_dc:
                fake_img = dc_projection(input_img, y_tilde, freq_mask)
            else:
                fake_img = y_tilde
            
            # Discriminator predictions
            if epoch >= config.warmup_epochs:
                disc_pred_fake = discriminator(fake_img)
                disc_pred_real = discriminator(target_img).detach()
            else:
                disc_pred_fake = None
                disc_pred_real = None
            
            # Generator loss
            loss_G, losses = criterion(fake_img, target_img, freq_mask, disc_pred_fake, disc_pred_real)
            
            loss_G.backward()
            optimizer_G.step()
            
            # ================== Train Discriminator ==================
            D_loss = torch.tensor(0.0)
            
            if epoch >= config.warmup_epochs:
                optimizer_D.zero_grad()
                
                # Real and fake predictions
                disc_real = discriminator(target_img)
                disc_fake = discriminator(fake_img.detach())
                
                # Relativistic average discriminator loss
                loss_D_real = F.mse_loss(disc_real - disc_fake.mean(), torch.ones_like(disc_real))
                loss_D_fake = F.mse_loss(disc_fake - disc_real.mean(), torch.zeros_like(disc_fake))
                D_loss = (loss_D_real + loss_D_fake) / 2
                
                D_loss.backward()
                optimizer_D.step()
            
            # 통계 업데이트
            epoch_losses['G'] += loss_G.item()
            epoch_losses['pix'] += losses['pix'].item()
            epoch_losses['perc'] += losses['perc'].item()
            epoch_losses['fft'] += losses['fft'].item()
            epoch_losses['adv'] += losses['adv'].item()
            epoch_losses['D'] += D_loss.item()
            
            # Progress bar
            pbar.set_postfix({
                'G': f"{loss_G.item():.4f}",
                'D': f"{D_loss.item():.4f}",
                'pix': f"{losses['pix'].item():.4f}",
                'fft': f"{losses['fft'].item():.4f}"
            })
        
        # Epoch 통계
        n = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= n
        
        print(f"Epoch {epoch+1} - G: {epoch_losses['G']:.4f}, "
              f"pix: {epoch_losses['pix']:.4f}, perc: {epoch_losses['perc']:.4f}, "
              f"fft: {epoch_losses['fft']:.4f}, adv: {epoch_losses['adv']:.4f}, "
              f"D: {epoch_losses['D']:.4f}")
        
        # 모델 저장
        if epoch_losses['G'] < best_loss:
            best_loss = epoch_losses['G']
            torch.save(generator.state_dict(), config.model_path_gen)
            torch.save(discriminator.state_dict(), config.model_path_disc)
            print(f"✓ Best model saved! Loss: {best_loss:.6f}")
        
        if (epoch + 1) % config.save_interval == 0:
            torch.save(generator.state_dict(), f'gen_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'disc_epoch_{epoch+1}.pth')
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    train()
