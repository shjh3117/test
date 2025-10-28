"""
Fast-SRGAN (MobileSRGAN) 모델 훈련
기존 ESPCN 데이터셋 구조를 활용한 GAN 훈련
"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.models as models

from FastSRGANconfig import fast_srgan_config as config
from FastSRGAN_models import FastSRGANGenerator, FastSRGANDiscriminator

class PerceptualLoss(nn.Module):
    """VGG 기반 Perceptual Loss"""
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        # VGG19의 특성 추출 레이어 사용
        try:
            vgg19 = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features
        except:
            # 이전 방식으로 폴백
            vgg19 = models.vgg19(pretrained=True).features
        
        # VGG19의 conv3_4 레이어까지만 사용 (경량화)
        self.feature_extractor = nn.Sequential(*list(vgg19.children())[:16])
        
        # 그래디언트 계산 비활성화
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.mse_loss = nn.MSELoss()
        
        # 정규화 (ImageNet 평균/표준편차)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """단일 채널을 RGB로 변환하고 정규화"""
        # 그레이스케일을 RGB로 변환
        x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        # 정규화
        pred_normalized = self.normalize(pred)
        target_normalized = self.normalize(target)
        
        # 특성 추출
        pred_features = self.feature_extractor(pred_normalized)
        target_features = self.feature_extractor(target_normalized)
        
        return self.mse_loss(pred_features, target_features)

class CombinedLoss(nn.Module):
    """Fast-SRGAN용 결합 손실함수"""
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.adversarial_loss = nn.BCELoss()
    
    def forward(self, pred, target, disc_pred_fake=None, is_discriminator=False):
        if is_discriminator:
            # Discriminator 손실만 계산
            return self.adversarial_loss(disc_pred_fake, torch.zeros_like(disc_pred_fake))
        
        # Generator 손실 계산
        content_loss = self.mse_loss(pred, target) + 0.1 * self.l1_loss(pred, target)
        
        perceptual_loss = self.perceptual_loss(pred, target) * config.perceptual_weight
        
        total_loss = content_loss * config.content_weight + perceptual_loss
        
        # Adversarial loss 추가 (warmup 기간 이후)
        if disc_pred_fake is not None:
            adversarial_loss = self.adversarial_loss(disc_pred_fake, torch.ones_like(disc_pred_fake))
            total_loss += adversarial_loss * config.adversarial_weight
        
        return total_loss

class FastSRGANDataset(Dataset):
    """Fast-SRGAN용 데이터셋 (기존 ESPCN 데이터 구조 활용)"""
    def __init__(self, augment=True):
        self.data_pairs = []
        self.augment = augment
        
        # 데이터 증강 transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-10, 10)),
            ])
        else:
            self.transform = None
        
        # work_dir 아래의 모든 비디오 폴더 찾기
        video_dirs = [d for d in os.listdir(config.work_dir) 
                      if os.path.isdir(os.path.join(config.work_dir, d))]
        
        for video_dir in video_dirs:
            target_dir = os.path.join(config.work_dir, video_dir, config.target_channel)
            input_dir = os.path.join(config.work_dir, video_dir, config.input_channel)
            
            if os.path.exists(target_dir) and os.path.exists(input_dir):
                target_files = sorted(glob.glob(os.path.join(target_dir, "*.png")))
                
                # 파일명이 매칭되는 쌍만 추가
                for target_file in target_files:
                    frame_name = os.path.basename(target_file)
                    input_file = os.path.join(input_dir, frame_name)
                    if os.path.exists(input_file):
                        self.data_pairs.append((input_file, target_file))
        
        print(f"Found {len(self.data_pairs)} training pairs for Fast-SRGAN")
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        input_path, target_path = self.data_pairs[idx]
        
        # 이미지 로드
        input_img = Image.open(input_path).convert('L')
        target_img = Image.open(target_path).convert('L')
        
        # 크기 확인 및 조정
        input_size = input_img.size
        target_size = target_img.size
        
        if input_size != target_size:
            # 타겟 크기에 맞춰 입력 이미지 리사이즈
            input_img = input_img.resize(target_size, Image.LANCZOS)
            print(f"Resized input from {input_size} to {target_size}")
        
        # PIL to Tensor
        input_tensor = transforms.ToTensor()(input_img)
        target_tensor = transforms.ToTensor()(target_img)
        
        # 데이터 증강 (동일한 변환을 input과 target에 적용)
        if self.transform:
            # 랜덤 시드 고정하여 같은 변환 적용
            seed = torch.randint(0, 2147483647, (1,)).item()
            
            torch.manual_seed(seed)
            input_tensor = self.transform(input_tensor)
            
            torch.manual_seed(seed)
            target_tensor = self.transform(target_tensor)
        
        # 정규화 (0~1 범위에서 -1~1 범위로)
        if config.normalize:
            input_tensor = input_tensor * 2.0 - 1.0
            target_tensor = target_tensor * 2.0 - 1.0
        
        return input_tensor, target_tensor

def train_fast_srgan():
    """Fast-SRGAN 훈련 함수"""
    # 디바이스 설정
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # FP16 설정
    use_amp = config.use_amp and device.type == 'cuda'
    if use_amp:
        print("Using Automatic Mixed Precision (FP16) training")
        scaler_gen = GradScaler('cuda')
        scaler_disc = GradScaler('cuda')
    else:
        print("Using FP32 training")
        scaler_gen = None
        scaler_disc = None
    
    # 데이터셋
    dataset = FastSRGANDataset(augment=True)
    if len(dataset) == 0:
        print("No training data found!")
        return
    
    # 데이터 분할
    train_size = int(config.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 모델 생성
    generator = FastSRGANGenerator().to(device)
    discriminator = FastSRGANDiscriminator().to(device)
    
    # T4 GPU 최적화
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # 손실함수
    criterion = CombinedLoss().to(device)
    
    # 옵티마이저
    optimizer_gen = optim.Adam(
        generator.parameters(), 
        lr=config.learning_rate_gen,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    optimizer_disc = optim.Adam(
        discriminator.parameters(), 
        lr=config.learning_rate_disc,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    # 스케줄러
    if config.scheduler == 'cosine':
        scheduler_gen = optim.lr_scheduler.CosineAnnealingLR(optimizer_gen, T_max=config.num_epochs)
        scheduler_disc = optim.lr_scheduler.CosineAnnealingLR(optimizer_disc, T_max=config.num_epochs)
    elif config.scheduler == 'step':
        scheduler_gen = optim.lr_scheduler.StepLR(optimizer_gen, step_size=config.num_epochs//3, gamma=0.1)
        scheduler_disc = optim.lr_scheduler.StepLR(optimizer_disc, step_size=config.num_epochs//3, gamma=0.1)
    else:
        scheduler_gen = None
        scheduler_disc = None
    
    best_val_loss = float('inf')
    
    # 훈련 루프
    for epoch in range(config.num_epochs):
        # 훈련 모드
        generator.train()
        discriminator.train()
        
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, (input_img, target_img) in enumerate(progress_bar):
            input_img = input_img.to(device, non_blocking=True)
            target_img = target_img.to(device, non_blocking=True)
            
            # =================== Generator 훈련 ===================
            optimizer_gen.zero_grad()
            
            if use_amp:
                with autocast(device_type='cuda'):
                    fake_img = generator(input_img)
                    
                    # Discriminator 예측 (warmup 기간 이후)
                    if epoch >= config.warmup_epochs:
                        disc_pred_fake = discriminator(fake_img.detach())
                        gen_loss = criterion(fake_img, target_img, disc_pred_fake, is_discriminator=False)
                    else:
                        gen_loss = criterion(fake_img, target_img, is_discriminator=False)
                
                scaler_gen.scale(gen_loss).backward()
                scaler_gen.step(optimizer_gen)
                scaler_gen.update()
            else:
                fake_img = generator(input_img)
                
                if epoch >= config.warmup_epochs:
                    disc_pred_fake = discriminator(fake_img.detach())
                    gen_loss = criterion(fake_img, target_img, disc_pred_fake, is_discriminator=False)
                else:
                    gen_loss = criterion(fake_img, target_img, is_discriminator=False)
                
                gen_loss.backward()
                optimizer_gen.step()
            
            gen_loss_total += gen_loss.item()
            
            # =================== Discriminator 훈련 ===================
            if epoch >= config.warmup_epochs and batch_idx % config.discriminator_update_freq == 0:
                optimizer_disc.zero_grad()
                
                if use_amp:
                    with autocast(device_type='cuda'):
                        # Real 이미지
                        disc_pred_real = discriminator(target_img)
                        disc_loss_real = criterion.adversarial_loss(
                            disc_pred_real, torch.ones_like(disc_pred_real)
                        )
                        
                        # Fake 이미지
                        fake_img_detached = fake_img.detach()
                        disc_pred_fake = discriminator(fake_img_detached)
                        disc_loss_fake = criterion.adversarial_loss(
                            disc_pred_fake, torch.zeros_like(disc_pred_fake)
                        )
                        
                        disc_loss = (disc_loss_real + disc_loss_fake) / 2
                    
                    scaler_disc.scale(disc_loss).backward()
                    scaler_disc.step(optimizer_disc)
                    scaler_disc.update()
                else:
                    # Real 이미지
                    disc_pred_real = discriminator(target_img)
                    disc_loss_real = criterion.adversarial_loss(
                        disc_pred_real, torch.ones_like(disc_pred_real)
                    )
                    
                    # Fake 이미지
                    fake_img_detached = fake_img.detach()
                    disc_pred_fake = discriminator(fake_img_detached)
                    disc_loss_fake = criterion.adversarial_loss(
                        disc_pred_fake, torch.zeros_like(disc_pred_fake)
                    )
                    
                    disc_loss = (disc_loss_real + disc_loss_fake) / 2
                    disc_loss.backward()
                    optimizer_disc.step()
                
                disc_loss_total += disc_loss.item()
            
            # Progress bar 업데이트
            progress_bar.set_postfix({
                'Gen Loss': f'{gen_loss.item():.4f}',
                'Disc Loss': f'{disc_loss.item() if epoch >= config.warmup_epochs else 0:.4f}'
            })
        
        # 검증
        if epoch % config.validate_interval == 0:
            generator.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for input_img, target_img in val_loader:
                    input_img, target_img = input_img.to(device), target_img.to(device)
                    
                    if use_amp:
                        with autocast(device_type='cuda'):
                            fake_img = generator(input_img)
                            loss = criterion.mse_loss(fake_img, target_img)
                    else:
                        fake_img = generator(input_img)
                        loss = criterion.mse_loss(fake_img, target_img)
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # 최고 모델 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(generator.state_dict(), config.model_path_gen)
                torch.save(discriminator.state_dict(), config.model_path_disc)
                print(f"Best models saved! Val loss: {best_val_loss:.6f}")
        
        # 에포크 정보 출력
        avg_gen_loss = gen_loss_total / len(train_loader)
        avg_disc_loss = disc_loss_total / max(len(train_loader) // config.discriminator_update_freq, 1)
        
        print(f"Epoch {epoch+1}: Gen={avg_gen_loss:.6f}, Disc={avg_disc_loss:.6f}")
        
        # 스케줄러 업데이트
        if scheduler_gen:
            scheduler_gen.step()
        if scheduler_disc:
            scheduler_disc.step()
        
        # 주기적 모델 저장
        if epoch % config.save_interval == 0:
            torch.save(generator.state_dict(), f'fast_srgan_generator_epoch_{epoch}.pth')
            torch.save(discriminator.state_dict(), f'fast_srgan_discriminator_epoch_{epoch}.pth')
    
    print("Fast-SRGAN training completed!")

if __name__ == "__main__":
    train_fast_srgan()