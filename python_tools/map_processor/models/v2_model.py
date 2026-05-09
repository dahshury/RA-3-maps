"""SegFormer-B0 + FiLM(style) backbone with multi-head outputs.

- 8-channel dense input (heightmap, water, 5 density, mp_spawn) replaces
  RGB. Patch-embed first conv is reinitialised: weights for channels 0-2
  copied from the pretrained RGB conv, channels 3-7 zero-init to a small
  random value. This preserves the ImageNet-pretrained early features
  while the new channels learn from scratch.
- All-MLP decoder per Xie et al. (NeurIPS 2021), with per-stage FiLM
  modulation conditioned on the DINOv2 style vector.
- Output heads: tile (palette_size), blend (present + secondary +
  direction), single_edge (same).

Per architecture_research_2026-05.md item 7. Reference impls:
  https://github.com/NVlabs/SegFormer
  transformers.SegformerModel ('nvidia/mit-b0')
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    palette_size: int
    n_directions: int = 17
    in_channels: int = 8
    style_dim: int = 256
    decoder_dim: int = 256
    blend_hidden: int = 128
    pretrained_name: str = "nvidia/mit-b0"


class _FiLM(nn.Module):
    """Per-channel scale + shift from style vector.

    v3: small Gaussian init (was zero-init) so style modulation has effect
    from step 1 — otherwise the dense input pathway dominates training and
    FiLM stays at no-op forever (the empirical failure mode of v2).
    """

    def __init__(self, style_dim: int, channels: int):
        super().__init__()
        self.fc = nn.Linear(style_dim, 2 * channels)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        sb = self.fc(style)
        scale, shift = sb.chunk(2, dim=-1)
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class _StyleCrossAttn(nn.Module):
    """Cross-attention from spatial features to a single style token.

    Queries = flattened spatial features. K/V = style projected to one token.
    Output residually added back to the input. The style now structurally
    *participates* in the decoder, not just modulating its scale/shift.
    """

    def __init__(self, channels: int, style_dim: int, n_heads: int = 4):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Linear(style_dim, 2 * channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.norm_q = nn.GroupNorm(min(8, channels), channels)
        self.gate = nn.Parameter(torch.zeros(1))   # learned residual scale
        nn.init.normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.kv.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W). style: (B, S).
        B, C, H, W = x.shape
        q_in = self.norm_q(x)
        q = self.q(q_in).flatten(2).transpose(1, 2)        # (B, HW, C)
        kv = self.kv(style)                                # (B, 2C)
        k, v = kv.chunk(2, dim=-1)
        k = k.unsqueeze(1)                                 # (B, 1, C)
        v = v.unsqueeze(1)
        out, _ = self.attn(q, k, v, need_weights=False)    # (B, HW, C)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + self.gate * out


class _StyledMLP(nn.Module):
    """SegFormer per-stage 1x1 conv head + FiLM + style cross-attention.

    v3 adds the cross-attn so the style condition is *bound* into each
    stage's features, not just scaled.
    """

    def __init__(self, in_dim: int, out_dim: int, style_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 1)
        self.cross = _StyleCrossAttn(out_dim, style_dim, n_heads=4)
        self.film = _FiLM(style_dim, out_dim)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.cross(h, style)
        return self.film(h, style)


class _BlendHead(nn.Module):
    """Predicts (present, secondary, direction) for one blend layer."""

    def __init__(self, in_ch: int, palette_size: int, n_directions: int,
                 hidden: int = 128, style_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + palette_size, hidden, 3, padding=1)
        self.film1 = _FiLM(style_dim, hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.film2 = _FiLM(style_dim, hidden)
        self.head_present = nn.Conv2d(hidden, 1, 1)
        self.head_secondary = nn.Conv2d(hidden, palette_size, 1)
        self.head_direction = nn.Conv2d(hidden, n_directions, 1)

    def forward(self, feat: torch.Tensor, tile_logits_detached: torch.Tensor,
                style: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Cascade: feed detached tile_logits as conditioning.
        x = torch.cat([feat, tile_logits_detached], dim=1)
        x = F.gelu(self.film1(self.conv1(x), style))
        x = F.gelu(self.film2(self.conv2(x), style))
        return {
            "present": self.head_present(x),
            "secondary": self.head_secondary(x),
            "direction": self.head_direction(x),
        }


class V2TextureNet(nn.Module):
    """SegFormer-B0 trunk + FiLM(style) MLP decoder + cascade tile/blend heads."""

    def __init__(self, palette_size: int, *, n_directions: int = 17,
                 in_channels: int = 8, style_dim: int = 256,
                 decoder_dim: int = 256, blend_hidden: int = 128,
                 pretrained_name: str = "nvidia/mit-b0"):
        super().__init__()
        from transformers import SegformerModel
        self.cfg = ModelConfig(
            palette_size=palette_size, n_directions=n_directions,
            in_channels=in_channels, style_dim=style_dim,
            decoder_dim=decoder_dim, blend_hidden=blend_hidden,
            pretrained_name=pretrained_name,
        )
        self.backbone = SegformerModel.from_pretrained(pretrained_name)
        self._adapt_input_conv(in_channels)
        hidden_sizes: List[int] = list(self.backbone.config.hidden_sizes)  # [32,64,160,256]
        # Per-stage MLP heads with FiLM conditioning.
        self.stage_heads = nn.ModuleList([
            _StyledMLP(hs, decoder_dim, style_dim) for hs in hidden_sizes
        ])
        # Fusion conv after concat.
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(hidden_sizes), decoder_dim, 1),
            nn.GELU(),
        )
        self.fuse_film = _FiLM(style_dim, decoder_dim)
        # Tile head (palette_size logits per pixel).
        self.tile_head = nn.Conv2d(decoder_dim, palette_size, 1)
        # Cascaded blend heads.
        self.blend_head = _BlendHead(decoder_dim, palette_size, n_directions,
                                     hidden=blend_hidden, style_dim=style_dim)
        self.single_head = _BlendHead(decoder_dim, palette_size, n_directions,
                                      hidden=blend_hidden, style_dim=style_dim)

    def _adapt_input_conv(self, in_channels: int) -> None:
        """Replace stage-0 patch_embed conv with one that takes `in_channels`.
        Init: copy pretrained RGB weights into channels 0..2; small random
        for the rest. Bias preserved.
        """
        old = self.backbone.encoder.patch_embeddings[0].proj
        new = nn.Conv2d(in_channels, old.out_channels,
                        kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, bias=(old.bias is not None))
        with torch.no_grad():
            new.weight.zero_()
            n_copy = min(in_channels, old.in_channels)
            new.weight[:, :n_copy] = old.weight[:, :n_copy]
            if in_channels > old.in_channels:
                # Small random init for novel channels (heightmap, water, ...).
                # Scale matches average abs of RGB weights.
                std = float(old.weight.std().item()) * 0.5
                new.weight[:, old.in_channels:].normal_(0.0, std)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.backbone.encoder.patch_embeddings[0].proj = new

    def encode(self, x_dense: torch.Tensor) -> List[torch.Tensor]:
        """x_dense: (B, in_channels, H, W) — H, W must be divisible by 32."""
        out = self.backbone(pixel_values=x_dense, output_hidden_states=True)
        return list(out.hidden_states)  # 4 multi-scale feature maps

    def decode(self, feats: List[torch.Tensor], style: torch.Tensor,
               out_hw: tuple[int, int]) -> torch.Tensor:
        """All-MLP decoder: project each stage to decoder_dim, upsample to
        target HW, concat, fuse."""
        H, W = out_hw
        target = (H // 4, W // 4)  # decode at 1/4 input res, then upsample.
        ups = []
        for h, head in zip(feats, self.stage_heads):
            y = head(h, style)
            if y.shape[-2:] != target:
                y = F.interpolate(y, size=target, mode="bilinear", align_corners=False)
            ups.append(y)
        z = self.fuse(torch.cat(ups, dim=1))
        z = self.fuse_film(z, style)
        # Upsample to full input resolution.
        z = F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)
        return z

    def forward(self, x_dense: torch.Tensor, style: torch.Tensor) -> Dict[str, torch.Tensor]:
        H, W = x_dense.shape[-2:]
        feats = self.encode(x_dense)
        z = self.decode(feats, style, (H, W))
        tile_logits = self.tile_head(z)
        # Cascade detached tile_logits (softmaxed) into blend heads.
        tile_cond = F.softmax(tile_logits.detach(), dim=1)
        blend = self.blend_head(z, tile_cond, style)
        single = self.single_head(z, tile_cond, style)
        return {"tiles": tile_logits, "blend": blend, "single_edge": single}

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
