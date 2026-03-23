import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CRNNModel(pl.LightningModule):
    def __init__(self, input_shape=(4, 257, 128), n_classes=37):
        super().__init__()

        # Conv Blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(8, 1))  # F:257→32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(4, 4))  # F:32→8 T:128→32
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(4, 4))  # F:8→2 T:32→8  (64,2,8)
        )

        # BiLSTM
        self.bilstm = nn.LSTM(input_size=64 * 2, hidden_size=64, bidirectional=True, batch_first=True) 

        self.classifier = nn.Sequential(
            nn.Flatten(),                                # [B, 8*128]
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)                    # [B, 37]
        )



    def forward(self, stft_cat, ild):
        # x: [B, C, F, T]
        x = self.conv1(stft_cat)  # → [B, 64, 32, 128]
        x = self.conv2(x)  # → [B, 64, 4, 128]
        x = self.conv3(x)  # → [B, 64, 2, 8]

        # reshape for LSTM: [B, C=64, F=2, T=8] → [B, T, 64*2]
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)  # [B, T=8, 128]

        x, _ = self.bilstm(x)  # [B, T, 128]
        x = self.classifier(x)        # [B, n_classes]

        return x
