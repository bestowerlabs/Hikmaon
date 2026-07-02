"""HikmaonNet — multi-branch deepfake detection network.

Design rationale (mirrors what the strongest published detectors combine):

- **Spatial branch** — a modern ConvNeXt-style backbone over RGB pixels.
  Learns semantic manipulation cues: blending boundaries, warping,
  inconsistent lighting, malformed anatomy.
- **Frequency branch** — a CNN over the log-magnitude 2-D FFT. Generative
  up-samplers (GAN transposed convs, diffusion decoders) leave periodic
  spectral fingerprints invisible in pixel space.
- **Noise branch** — fixed SRM high-pass filters feeding a small CNN.
  Splices and synthesis disturb the camera-sensor noise pattern; SRM
  residuals are the classic forensic input for detecting that.
- **Attention fusion** — branch tokens fused by multi-head self-attention,
  so the network learns *which* evidence stream to trust per input.
- **Calibration** — a learnable temperature applied at inference so the
  output is a usable probability, not just a ranking score (fit it with
  `evaluate.fit_temperature` on a held-out split after training).

Input: 224x224 RGB, ImageNet-normalized (see ml/data.py).
Output: dict with `logit`, `probability` (temperature-scaled).

Training tips for the team:
- Balance real/fake per batch; group val/test splits by *generator* to
  measure cross-generator generalization (the metric that matters).
- Keep the heavy JPEG/resize augmentations ON (ml/data.py) — detectors
  without them collapse the moment content is re-shared and recompressed.
- Start with `--freeze-epochs 1` if fine-tuning from a checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_VERSION = "hikmaonnet-v1"
INPUT_SIZE = 224


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise 7x7 -> LN -> pointwise MLP with residual."""

    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = (self.gamma * x).permute(0, 3, 1, 2)
        if self.training and self.drop_path > 0:
            keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.drop_path
            x = x * keep / (1 - self.drop_path)
        return residual + x


class Stage(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, depth: int, downsample: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if downsample:
            layers += [
                nn.GroupNorm(1, dim_in),
                nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2),
            ]
        elif dim_in != dim_out:
            layers.append(nn.Conv2d(dim_in, dim_out, kernel_size=1))
        layers += [ConvNeXtBlock(dim_out) for _ in range(depth)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SpatialBranch(nn.Module):
    """ConvNeXt-style RGB backbone. dims/depths sized for single-GPU training;
    scale up dims=(128,256,512,1024) when the team has A100-class hardware."""

    def __init__(self, out_dim: int, dims=(64, 128, 256, 512), depths=(2, 2, 6, 2)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0]),
        )
        self.stages = nn.ModuleList(
            [
                Stage(dims[max(i - 1, 0)], dims[i], depths[i], downsample=i > 0)
                for i in range(len(dims))
            ]
        )
        self.head = nn.Linear(dims[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = x.mean(dim=(2, 3))  # global average pool
        return self.head(x)


class FrequencyBranch(nn.Module):
    """CNN over log-magnitude FFT of the grayscale image."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.GELU(), nn.GroupNorm(1, 32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(), nn.GroupNorm(1, 64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(), nn.GroupNorm(1, 128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, out_dim)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        gray = rgb.mean(dim=1, keepdim=True)
        spectrum = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"), dim=(-2, -1))
        magnitude = torch.log1p(spectrum.abs())
        magnitude = (magnitude - magnitude.mean(dim=(-2, -1), keepdim=True)) / (
            magnitude.std(dim=(-2, -1), keepdim=True) + 1e-6
        )
        features = self.net(magnitude).flatten(1)
        return self.head(features)


def _srm_kernels() -> torch.Tensor:
    """Three canonical SRM high-pass filters (steganalysis / forensics)."""
    k1 = torch.tensor(
        [[0, 0, 0, 0, 0],
         [0, -1, 2, -1, 0],
         [0, 2, -4, 2, 0],
         [0, -1, 2, -1, 0],
         [0, 0, 0, 0, 0]], dtype=torch.float32) / 4.0
    k2 = torch.tensor(
        [[-1, 2, -2, 2, -1],
         [2, -6, 8, -6, 2],
         [-2, 8, -12, 8, -2],
         [2, -6, 8, -6, 2],
         [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
    k3 = torch.zeros(5, 5)
    k3[2, 1], k3[2, 2], k3[2, 3] = 1.0, -2.0, 1.0
    k3 = k3 / 2.0
    kernels = torch.stack([k1, k2, k3]).unsqueeze(1)  # (3,1,5,5)
    return kernels.repeat(1, 3, 1, 1) / 3.0  # apply across RGB


class NoiseBranch(nn.Module):
    """Fixed SRM residual extraction followed by a learnable CNN."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.register_buffer("srm", _srm_kernels())
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(), nn.GroupNorm(1, 32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(), nn.GroupNorm(1, 64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, out_dim)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        residual = F.conv2d(rgb, self.srm, padding=2)
        features = self.net(residual).flatten(1)
        return self.head(features)


class AttentionFusion(nn.Module):
    """Self-attention over branch tokens + learned CLS token."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        fused, _ = self.attn(sequence, sequence, sequence)
        return self.norm(fused[:, 0])


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class HikmaonNet(nn.Module):
    def __init__(self, embed_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.spatial = SpatialBranch(embed_dim)
        self.frequency = FrequencyBranch(embed_dim)
        self.noise = NoiseBranch(embed_dim)
        self.fusion = AttentionFusion(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        # Calibration temperature — fit post-training on a held-out split.
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = torch.stack(
            [self.spatial(x), self.frequency(x), self.noise(x)], dim=1
        )
        fused = self.fusion(tokens)
        logit = self.classifier(fused).squeeze(-1)
        return {
            "logit": logit,
            "probability": torch.sigmoid(logit / self.temperature.clamp(min=1e-3)),
        }


def build_model(embed_dim: int = 256, dropout: float = 0.3) -> HikmaonNet:
    return HikmaonNet(embed_dim=embed_dim, dropout=dropout)
