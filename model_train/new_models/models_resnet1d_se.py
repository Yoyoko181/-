from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm1d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = x.transpose(1, 2)
        return x


class SEBlock1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class MultiScaleConv1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernels: Tuple[int, int, int] = (3, 5, 7),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        k3, k5, k7 = kernels
        self.b3 = nn.Conv1d(in_ch, out_ch, kernel_size=k3, padding=k3 // 2, bias=False)
        self.b5 = nn.Conv1d(in_ch, out_ch, kernel_size=k5, padding=k5 // 2, bias=False)
        self.b7 = nn.Conv1d(in_ch, out_ch, kernel_size=k7, padding=k7 // 2, bias=False)
        self.fuse = nn.Conv1d(out_ch * 3, out_ch, kernel_size=1, bias=False)
        self.norm = LayerNorm1d(out_ch)
        self.se = SEBlock1d(out_ch)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.b3(x), self.b5(x), self.b7(x)], dim=1)
        x = self.fuse(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        x = self.se(x)
        x = self.drop(x)
        return x


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = LayerNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = LayerNorm1d(out_ch)
        self.se = SEBlock1d(out_ch)
        self.drop = nn.Dropout(p=dropout)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                LayerNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class ResNet1DSE(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 64,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            LayerNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.ms = MultiScaleConv1d(base_channels, base_channels, dropout=dropout)

        ch = base_channels
        self.layer1, ch = self._make_layer(ch, base_channels, layers[0], stride=1, dropout=dropout)
        self.layer2, ch = self._make_layer(ch, base_channels * 2, layers[1], stride=2, dropout=dropout)
        self.layer3, ch = self._make_layer(ch, base_channels * 4, layers[2], stride=2, dropout=dropout)
        self.layer4, ch = self._make_layer(ch, base_channels * 8, layers[3], stride=2, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ch, num_classes)

    def _make_layer(
        self, in_ch: int, out_ch: int, blocks: int, stride: int, dropout: float
    ) -> Tuple[nn.Sequential, int]:
        layers: List[nn.Module] = []
        layers.append(BasicBlock1d(in_ch, out_ch, stride=stride, dropout=dropout))
        for _ in range(1, blocks):
            layers.append(BasicBlock1d(out_ch, out_ch, stride=1, dropout=dropout))
        return nn.Sequential(*layers), out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.ms(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

