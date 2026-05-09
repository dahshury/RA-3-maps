"""Style-conditioned U-Net for per-tile texture classification.

Inputs:
    x         : (B, C_in, H, W) float - terrain/object/mask channels
    style_id  : (B,) long - cluster id [0..n_styles)

Output:
    logits    : (B, vocab_size, H, W) float

Style conditioning:
    style_id -> learned embedding -> broadcast HxW and concatenated to the input.
    Additional FiLM-style modulation (per-channel scale+shift) at the bottleneck.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if size mismatch from odd dims
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh or dw:
            x = F.pad(x, [0, dw, 0, dh])
        return self.conv(torch.cat([skip, x], dim=1))


class StyleFiLM(nn.Module):
    """Per-channel scale+shift conditioned on style embedding."""

    def __init__(self, style_dim: int, n_channels: int):
        super().__init__()
        self.fc = nn.Linear(style_dim, 2 * n_channels)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.n_channels = n_channels

    def forward(self, x, style_emb):
        gamma_beta = self.fc(style_emb)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        # x: (B,C,H,W); gamma,beta: (B,C). Broadcast to spatial dims.
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class TextureTransferUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        n_styles: int = 8,
        style_dim: int = 16,
        vocab_size: int = 350,
        base: int = 32,
    ):
        super().__init__()
        self.style_emb = nn.Embedding(n_styles, style_dim)

        self.in_conv = _ConvBlock(in_channels + style_dim, base)
        self.down1 = _Down(base, base * 2)
        self.down2 = _Down(base * 2, base * 4)
        self.down3 = _Down(base * 4, base * 8)
        self.bottleneck = _ConvBlock(base * 8, base * 8)
        self.bottleneck_film = StyleFiLM(style_dim, base * 8)

        self.up3 = _Up(base * 8, base * 4, base * 4)
        self.up2 = _Up(base * 4, base * 2, base * 2)
        self.up1 = _Up(base * 2, base, base)
        self.out_conv = nn.Conv2d(base, vocab_size, 1)

    def forward(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        s = self.style_emb(style_id)  # (B, style_dim)
        # Broadcast style embedding spatially as extra input channels
        s_map = s[:, :, None, None].expand(-1, -1, H, W)
        x = torch.cat([x, s_map], dim=1)

        x1 = self.in_conv(x)        # (B, base, H, W)
        x2 = self.down1(x1)         # (B, 2b, H/2, W/2)
        x3 = self.down2(x2)         # (B, 4b, H/4, W/4)
        x4 = self.down3(x3)         # (B, 8b, H/8, W/8)
        b = self.bottleneck(x4)
        b = self.bottleneck_film(b, s)

        u3 = self.up3(b, x3)
        u2 = self.up2(u3, x2)
        u1 = self.up1(u2, x1)
        return self.out_conv(u1)
