import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class CPNet(pl.LightningModule):
    def __init__(self, in_channels=1, n_classes=37):
        super().__init__()

        # IPD input shape: (B, 1, F, T)
        self.conv1 = DepthwiseSeparableConv(in_channels, 8, kernel_size=3)
        self.conv2 = DepthwiseSeparableConv(8, 16, kernel_size=3)
        self.conv3 = DepthwiseSeparableConv(16, 32, kernel_size=3)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Flatten to feed into GRU: (B, T, F) style
        self.gru = nn.GRU(input_size=32, hidden_size=32, num_layers=1, 
                          batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(16 * 32, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.conv1(x)   # (B, 8, F, T)
        x = self.conv2(x)   # (B, 16, F, T)
        x = self.conv3(x)   # (B, 32, F, T)

        x = self.pool(x)    # (B, 32, 32, 32)
        x = x.permute(0, 3, 1, 2).contiguous().view(x.size(0), 32, -1)  # (B, T=32, F=32)

        #x, _ = self.gru(x)  # (B, 32, 64)
        x = x.contiguous().view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
class CPGruNet2(pl.LightningModule):
    def __init__(self, in_channels=1, n_classes=37):
        super().__init__()

        # IPD input shape: (B, 1, F, T)
        self.conv1 = DepthwiseSeparableConv(in_channels, 8, kernel_size=3)
        self.conv2 = DepthwiseSeparableConv(8, 16, kernel_size=3)
        self.conv3 = DepthwiseSeparableConv(16, 32, kernel_size=3)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Flatten to feed into GRU: (B, T, F) style
        self.gru = nn.GRU(input_size=4, hidden_size=8, num_layers=1, 
                          batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(16 * 32, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.conv1(x)   # (B, 8, F, T)
        x = self.conv2(x)   # (B, 16, F, T)
        x = self.conv3(x)   # (B, 32, F, T)

        x = self.pool(x)    # (B, 32, 32, 32)
        x = x.permute(0, 3, 1, 2).contiguous().view(x.size(0), 32, -1)  # (B, T=32, F=32)

        #x, _ = self.gru(x)  # (B, 32, 64)
        x = x.contiguous().view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x        
class CPGruNet3(pl.LightningModule):
    def __init__(self, in_channels=1, n_classes=37):
        super().__init__()

        # IPD input shape: (B, 1, F, T)
        self.conv1 = DepthwiseSeparableConv(in_channels, 8, kernel_size=3)
        self.conv2 = DepthwiseSeparableConv(8, 16, kernel_size=3)
        self.conv3 = DepthwiseSeparableConv(16, 32, kernel_size=3)

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # Flatten to feed into GRU: (B, T, F) style
        self.gru = nn.GRU(input_size=4, hidden_size=4, num_layers=1, 
                          batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(8 * 32, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.conv1(x)   # (B, 8, F, T)
        x = self.conv2(x)   # (B, 16, F, T)
        x = self.conv3(x)   # (B, 32, F, T)

        x = self.pool(x)    # (B, 32, 32, 32)
        x = x.permute(0, 3, 1, 2).contiguous().view(x.size(0), 32, -1)  # (B, T=32, F=32)

        x, _ = self.gru(x)  # (B, 32, 64)
        x = x.contiguous().view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x          
class CPGruNet4(pl.LightningModule):
    def __init__(self, in_channels=1, n_classes=37):
        super().__init__()

        # IPD input shape: (B, 1, F, T)
        self.conv1 = DepthwiseSeparableConv(in_channels, 8, kernel_size=3)
        self.conv2 = DepthwiseSeparableConv(8, 16, kernel_size=3)
        self.conv3 = DepthwiseSeparableConv(16, 32, kernel_size=3)

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # Flatten to feed into GRU: (B, T, F) style
        self.gru = nn.GRU(input_size=4, hidden_size=8, num_layers=1, 
                          batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(16 * 32, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.conv1(x)   # (B, 8, F, T)
        x = self.conv2(x)   # (B, 16, F, T)
        x = self.conv3(x)   # (B, 32, F, T)

        x = self.pool(x)    # (B, 32, 32, 32)
        x = x.permute(0, 3, 1, 2).contiguous().view(x.size(0), 32, -1)  # (B, T=32, F=32)

        x, _ = self.gru(x)  # (B, 32, 64)
        x = x.contiguous().view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
class CPGruNet5(pl.LightningModule):
    def __init__(self, in_channels=1, n_classes=37):
        super().__init__()

        # IPD input shape: (B, 1, F, T)
        self.conv1 = DepthwiseSeparableConv(in_channels, 8, kernel_size=3)
        self.conv2 = DepthwiseSeparableConv(8, 16, kernel_size=3)
        self.conv3 = DepthwiseSeparableConv(16, 32, kernel_size=3)

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # Flatten to feed into GRU: (B, T, F) style
        self.gru = nn.GRU(input_size=4, hidden_size=8, num_layers=1, 
                          batch_first=True, bidirectional=False)

        self.fc1 = nn.Linear(8 * 32, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.conv1(x)   # (B, 8, F, T)
        x = self.conv2(x)   # (B, 16, F, T)
        x = self.conv3(x)   # (B, 32, F, T)

        x = self.pool(x)    # (B, 32, 32, 32)
        x = x.permute(0, 3, 1, 2).contiguous().view(x.size(0), 32, -1)  # (B, T=32, F=32)

        x, _ = self.gru(x)  # (B, 32, 64)
        x = x.contiguous().view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x         
class CPGruNet(pl.LightningModule):
    def __init__(self, in_channels=1, n_classes=37):
        super().__init__()

        # IPD input shape: (B, 1, F, T)
        self.conv1 = DepthwiseSeparableConv(in_channels, 8, kernel_size=3)
        self.conv2 = DepthwiseSeparableConv(8, 16, kernel_size=3)
        self.conv3 = DepthwiseSeparableConv(16, 32, kernel_size=3)

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # Flatten to feed into GRU: (B, T, F) style
        self.gru = nn.GRU(input_size=2, hidden_size=2, num_layers=1, 
                          batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(4 * 32, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.conv1(x)   # (B, 8, F, T)
        x = self.conv2(x)   # (B, 16, F, T)
        x = self.conv3(x)   # (B, 32, F, T)

        x = self.pool(x)    # (B, 32, 32, 32)
        x = x.permute(0, 3, 1, 2).contiguous().view(x.size(0), 32, -1)  # (B, T=32, F=32)

        #x, _ = self.gru(x)  # (B, 32, 64)
        x = x.contiguous().view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
