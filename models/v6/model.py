# Model Class
# **↑このコメントは残すこと**
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class DrumResNet(nn.Module):
    def __init__(self, num_classes=16, n_mels=128):
        super(DrumResNet, self).__init__()
        self.n_mels = n_mels
        
        # ---------------------------------------------------------
        # Branch 1: Time-Normalized (ADSR・アタック形状の学習)
        # ---------------------------------------------------------
        self.time_norm_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualBlock2d(16, 32, stride=2),
            ResidualBlock2d(32, 64, stride=2),
            ResidualBlock2d(64, 32, stride=2)
        )
        
        # ---------------------------------------------------------
        # Branch 2: Bin-Normalized (音色・周波数バランスの学習)
        # ---------------------------------------------------------
        self.bin_norm_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualBlock2d(16, 32, stride=2),
            ResidualBlock2d(32, 32, stride=2),
            ResidualBlock2d(32, 16, stride=2)
        )
        
        self.fc1 = nn.Linear(32*4+16*4, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        
        # (Batch, 1, n_mels, Time)に復元
        x_2d = x.view(batch_size, 1, self.n_mels, -1)
        
        # --- 動的正規化 (Dynamic Normalization) ---
        
        # 1. Time軸方向の正規化 (dim=-1)
        mean_time = x_2d.mean(dim=-1, keepdim=True)
        std_time = x_2d.std(dim=-1, keepdim=True)
        x_time_norm = (x_2d - mean_time) / (std_time + 1e-6)
        
        # 2. Bin(周波数)軸方向の正規化 (dim=2)
        mean_bin = x_2d.mean(dim=2, keepdim=True)
        std_bin = x_2d.std(dim=2, keepdim=True)
        x_bin_norm = (x_2d - mean_bin) / (std_bin + 1e-6)
        
        # --- 特徴抽出 ---
        
        # Branch 1
        feat_time = self.time_norm_branch(x_time_norm)
        # 時間軸(X)のみを1に圧縮し、周波数軸(Y)の4帯域(Low, Low-Mid, High-Mid, High)を保持する
        feat_time = F.adaptive_max_pool2d(feat_time, (4, 1))
        feat_time = feat_time.view(batch_size, -1)
        
        # Branch 2
        feat_bin = self.bin_norm_branch(x_bin_norm)
        # こちらも周波数帯域を保持
        feat_bin = F.adaptive_max_pool2d(feat_bin, (4, 1))
        feat_bin = feat_bin.view(batch_size, -1)
        
        # --- 結合と分類 ---
        combined_feat = torch.cat((feat_time, feat_bin), dim=1)
        #combined_feat = feat_time + feat_bin
        out = self.fc1(combined_feat)
        #out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out