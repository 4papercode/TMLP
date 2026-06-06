import torch
import torch.nn as nn


class PI_FeatureExtractor(nn.Module):
    """
    CNN-based Persistence Image feature extractor.
    Input:  (B, 1, 10, 10)  — single-channel 10x10 PI image
    Output: (B, feature_dim) — topology embedding
    """
    def __init__(self, feature_dim=128):
        super(PI_FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # (16, 10, 10)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)              # (32, 8, 8)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(kernel_size=2)                   # (32, 4, 4)

        self.fc_extract = nn.Linear(32 * 4 * 4, feature_dim)      # 512 → 128
        self.dropout    = nn.Dropout(0.3)

    def forward(self, x):
        # x: (B, 1, 10, 10)
        x = self.relu(self.bn1(self.conv1(x)))   # (B, 16, 10, 10)
        x = self.relu(self.bn2(self.conv2(x)))   # (B, 32, 8, 8)
        x = self.pool(x)                         # (B, 32, 4, 4)
        x = torch.flatten(x, 1)                  # (B, 512)
        x = self.relu(self.fc_extract(x))        # (B, 128)
        x = self.dropout(x)
        return x                                 # (B, feature_dim=128)
