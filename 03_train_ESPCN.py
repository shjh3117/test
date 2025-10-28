"""
ESPCN (Efficient Sub-Pixel CNN) 모델 훈련
ESPCNconfig.py를 사용하여 하이퍼파라미터 관리
"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
from ESPCNconfig import config

class ESPCN(nn.Module):
    def __init__(self):
        super(ESPCN, self).__init__()
        
        # 설정에서 파라미터 가져오기
        num_features = config.num_features
        num_layers = config.num_layers
        kernel_size = config.kernel_size
        use_bn = config.use_batch_norm
        dropout_rate = config.dropout_rate
        
        # Sequential 모델로 간소화
        layers = []
        
        # 첫 번째 레이어
        layers.append(nn.Conv2d(1, num_features, kernel_size=kernel_size, padding=kernel_size//2))
        
        if config.activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        
        # 중간 레이어들
        for i in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            
            if config.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))
            
            if use_bn:
                layers.append(nn.BatchNorm2d(num_features))
            
            if dropout_rate > 0.0 and i % 2 == 1:
                layers.append(nn.Dropout2d(dropout_rate))
        
        # 마지막 레이어
        layers.append(nn.Conv2d(num_features, 1, kernel_size=3, padding=1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# 간단한 Combined Loss (MSE + L1)
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.mse_weight = config.mse_weight
        self.l1_weight = config.l1_weight
    
    def forward(self, pred, target):
        return (self.mse_weight * self.mse_loss(pred, target) + 
                self.l1_weight * self.l1_loss(pred, target))

class VideoDataset(Dataset):
    def __init__(self):
        self.data_pairs = []
        
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
        
        print(f"Found {len(self.data_pairs)} training pairs")
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        input_path, target_path = self.data_pairs[idx]
        
        # 이미지 로드
        input_img = Image.open(input_path).convert('L')
        target_img = Image.open(target_path).convert('L')
        
        # numpy 배열로 변환
        input_array = np.array(input_img, dtype=np.float32)
        target_array = np.array(target_img, dtype=np.float32)
        
        # 정규화
        if config.normalize:
            input_array = input_array / 255.0
            target_array = target_array / 255.0
        
        # 텐서로 변환 (C, H, W)
        input_tensor = torch.from_numpy(input_array).unsqueeze(0)
        target_tensor = torch.from_numpy(target_array).unsqueeze(0)
        
        return input_tensor, target_tensor

def train_model():
    # 디바이스 설정
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # 데이터셋
    dataset = VideoDataset()
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
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    # 모델 및 손실함수
    model = ESPCN().to(device)
    criterion = CombinedLoss()
    
    # 옵티마이저 설정
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, 
                              weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, 
                             weight_decay=config.weight_decay, momentum=0.9)
    else:  # adam
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                              weight_decay=config.weight_decay)
    
    # 스케줄러 설정
    if config.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.scheduler_patience, 
            factor=config.scheduler_factor
        )
    elif config.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    elif config.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.num_epochs//3, gamma=0.1)
    else:
        scheduler = None
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # 훈련
        model.train()
        train_loss = 0.0
        for batch_idx, (input_img, target_img) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_img, target_img = input_img.to(device), target_img.to(device)
            
            optimizer.zero_grad()
            pred_img = model(input_img)
            loss = criterion(pred_img, target_img)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_img, target_img in val_loader:
                input_img, target_img = input_img.to(device), target_img.to(device)
                pred_img = model(input_img)
                val_loss += criterion(pred_img, target_img).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
        
        # 스케줄러 업데이트
        if scheduler:
            if config.scheduler == 'plateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # 최고 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.model_path)
            print(f"Best model saved! Val loss: {best_val_loss:.6f}")
    
    print("Training completed!")

if __name__ == "__main__":
    train_model()