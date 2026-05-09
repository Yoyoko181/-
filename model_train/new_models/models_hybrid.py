from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from new_models.models_resnet1d_se import LayerNorm1d, MultiScaleConv1d, SEBlock1d


class BasicBlock1dOpt(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dropout: float = 0.25,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = LayerNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = LayerNorm1d(out_ch)
        self.se = SEBlock1d(out_ch) if use_se else nn.Identity()
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


class ResNet1DBackbone(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout: float = 0.25,
        use_multiscale: bool = True,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            LayerNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.ms = (
            MultiScaleConv1d(base_channels, base_channels, dropout=dropout)
            if use_multiscale
            else nn.Identity()
        )

        ch = base_channels
        self.layer1, ch = self._make_layer(
            ch, base_channels, layers[0], stride=1, dropout=dropout, use_se=use_se
        )
        self.layer2, ch = self._make_layer(
            ch, base_channels * 2, layers[1], stride=2, dropout=dropout, use_se=use_se
        )
        self.layer3, ch = self._make_layer(
            ch, base_channels * 4, layers[2], stride=2, dropout=dropout, use_se=use_se
        )
        self.layer4, ch = self._make_layer(
            ch, base_channels * 8, layers[3], stride=2, dropout=dropout, use_se=use_se
        )
        self.out_channels = ch

    def _make_layer(
        self, in_ch: int, out_ch: int, blocks: int, stride: int, dropout: float, use_se: bool
    ):
        seq = []
        seq.append(
            BasicBlock1dOpt(in_ch, out_ch, stride=stride, dropout=dropout, use_se=use_se)
        )
        for _ in range(1, blocks):
            seq.append(
                BasicBlock1dOpt(out_ch, out_ch, stride=1, dropout=dropout, use_se=use_se)
            )
        return nn.Sequential(*seq), out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.ms(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class HybridResNetBiLSTMAttn(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 64,
        dropout: float = 0.25,
        use_multiscale: bool = True,
        use_se: bool = True,
        use_lstm: bool = True,
        use_attn: bool = True,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = ResNet1DBackbone(
            base_channels=base_channels,
            dropout=dropout,
            use_multiscale=use_multiscale,
            use_se=use_se,
        )
        self.use_lstm = use_lstm
        self.use_attn = use_attn and use_lstm

        c = self.backbone.out_channels
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=c,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                dropout=dropout if lstm_layers > 1 else 0.0,
                bidirectional=True,
                batch_first=True,
            )
            d = lstm_hidden * 2
            self.ln_seq = nn.LayerNorm(d)
            if self.use_attn:
                self.attn = nn.MultiheadAttention(
                    embed_dim=d,
                    num_heads=attn_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            else:
                self.attn = None
            feat_dim = d * 2
        else:
            self.lstm = None
            self.attn = None
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.pool_max = nn.AdaptiveMaxPool1d(1)
            feat_dim = c * 2

        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if not self.use_lstm:
            avg = self.pool(x).squeeze(-1)
            mx = self.pool_max(x).squeeze(-1)
            feat = torch.cat([avg, mx], dim=1)
            return self.head(feat)

        seq = x.transpose(1, 2)
        seq, _ = self.lstm(seq)
        seq = self.ln_seq(seq)
        seq = F.relu(seq, inplace=True)
        if self.attn is not None:
            seq, _ = self.attn(seq, seq, seq, need_weights=False)
        mean_pool = seq.mean(dim=1)
        max_pool = seq.max(dim=1).values
        feat = torch.cat([mean_pool, max_pool], dim=1)
        return self.head(feat)


@dataclass(frozen=True)
class HybridVariant:
    use_multiscale: bool
    use_se: bool
    use_lstm: bool
    use_attn: bool


def hybrid_variant(name: str) -> HybridVariant:
    name = (name or "").strip().lower()
    if name in ("base", "resnet"):
        return HybridVariant(use_multiscale=False, use_se=False, use_lstm=False, use_attn=False)
    if name in ("ms", "multiscale"):
        return HybridVariant(use_multiscale=True, use_se=False, use_lstm=False, use_attn=False)
    if name in ("ms_se", "multiscale_se"):
        return HybridVariant(use_multiscale=True, use_se=True, use_lstm=False, use_attn=False)
    if name in ("ms_se_lstm",):
        return HybridVariant(use_multiscale=True, use_se=True, use_lstm=True, use_attn=False)
    if name in ("full", "ms_se_lstm_attn"):
        return HybridVariant(use_multiscale=True, use_se=True, use_lstm=True, use_attn=True)
    raise ValueError(f"Unknown hybrid_variant={name}")

