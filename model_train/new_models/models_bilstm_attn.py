from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttnClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 1,
        d_model: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        attn_heads: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=False,
        )

        self.out = nn.Sequential(
            nn.Linear(lstm_hidden * 4, lstm_hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.ln(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)

        x, _ = self.lstm(x)

        qkv = x.transpose(0, 1)
        attn_out, _ = self.attn(qkv, qkv, qkv, need_weights=False)
        attn_out = attn_out.transpose(0, 1)

        mean_pool = attn_out.mean(dim=1)
        max_pool = attn_out.max(dim=1).values
        feat = torch.cat([mean_pool, max_pool], dim=1)
        logits = self.out(feat)
        return logits

