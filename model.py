"""
model.py — SwinEVDNet3D architecture for brain tumor segmentation.

Copied verbatim from training notebook swin_evdnet_v2_native_3d.ipynb,
cells 6 (model definition, lines 1941-2153). Only removed:
  - training-only imports and notebook magics
  - GRAD_CHECKPOINTING / ENCODER_3D / DEVICE globals (hardcoded below for inference)
  - CLS_HIDDEN / CLS_DROPOUT / NUM_CLASSES globals (passed as defaults below)

Layer names, shapes, and constructor arguments are IDENTICAL to training.

Checkpoint loading:
    ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)

Constructor used during training:
    model = SwinEVDNet3D(num_classes=3, encoder="swin")
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinTransformer

# ── Training config values (fixed; never changed between training and inference) ──
FEATURE_SIZE  = 24
SWIN_DEPTHS   = (2, 2, 2, 2)
SWIN_HEADS    = (3, 6, 12, 24)
SWIN_WINDOW   = 7
DROP_PATH     = 0.1
ASPP_CH       = 128
DECODER_CH    = (96, 64, 48, 32)
CLS_HIDDEN    = 256
CLS_DROPOUT   = 0.3


def norm3d(c: int) -> nn.Module:
    """GroupNorm: independent of batch size and valid on 1-voxel maps."""
    return nn.GroupNorm(next(g for g in (8, 4, 2, 1) if c % g == 0), c)


class ConvNormAct3d(nn.Sequential):
    """Conv3d → GroupNorm → LeakyReLU. Supports groups for depthwise convs."""
    def __init__(self, cin, cout, k=3, stride=1, dilation=1, groups=1):
        super().__init__(
            nn.Conv3d(cin, cout, k, stride,
                      padding=dilation * (k // 2),
                      dilation=dilation, groups=groups, bias=False),
            norm3d(cout),
            nn.LeakyReLU(0.01, inplace=True),
        )


class ResBlock3d(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = ConvNormAct3d(cin, cout, 3, stride)
        self.c2 = nn.Sequential(
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), norm3d(cout))
        self.skip = (nn.Identity() if cin == cout and stride == 1
                     else nn.Sequential(
                         nn.Conv3d(cin, cout, 1, stride, bias=False), norm3d(cout)))
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.c2(self.c1(x)) + self.skip(x))


class SwinEncoder3D(nn.Module):
    """MONAI 3D SwinTransformer with training-matched constructor args."""
    def __init__(self, in_ch: int, F: int):
        super().__init__()
        import inspect
        kw = dict(
            in_chans=in_ch, embed_dim=F,
            window_size=(SWIN_WINDOW,) * 3, patch_size=(2, 2, 2),
            depths=tuple(SWIN_DEPTHS), num_heads=tuple(SWIN_HEADS),
            mlp_ratio=4.0, qkv_bias=True,
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=DROP_PATH,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,   # only True during training (GRAD_CHECKPOINTING)
            spatial_dims=3,
        )
        accepted = inspect.signature(SwinTransformer.__init__).parameters
        self.swin = SwinTransformer(**{k: v for k, v in kw.items() if k in accepted})
        self.channels = [F * m for m in (1, 2, 4, 8, 16)]
        self.kind = "monai-SwinTransformer-3D"

    def set_grad_checkpointing(self, enable: bool) -> bool:
        n = 0
        for m in self.swin.modules():
            if hasattr(m, "use_checkpoint"):
                m.use_checkpoint = bool(enable)
                n += 1
        return n > 0

    def forward(self, x):
        return list(self.swin(x, normalize=True))


class ASPP3D(nn.Module):
    """Depthwise-separable dilated 3D convs, rates [1,2,3] + image pooling → 1×1 projection."""
    def __init__(self, cin: int, cout: int, rates=(1, 2, 3)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                ConvNormAct3d(cin, cin, 3, dilation=r, groups=cin),  # depthwise
                ConvNormAct3d(cin, cout, 1),                          # pointwise
            )
            for r in rates
        ])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Conv3d(cin, cout, 1),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.project = ConvNormAct3d(cout * (len(rates) + 1), cout, 1)

    def forward(self, x):
        feats = [b(x) for b in self.branches] + [
            self.pool(x).expand(-1, -1, *x.shape[-3:])]
        return self.project(torch.cat(feats, 1))


class UpBlock3d(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.block = ResBlock3d(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear",
                          align_corners=False)
        return self.block(torch.cat([x, skip], 1))


class EvidentialSegHead3D(nn.Module):
    """Voxel-wise Dirichlet head: returns (alpha, prob, unc)."""
    def __init__(self, cin: int, K: int = 2):
        super().__init__()
        self.conv, self.K = nn.Conv3d(cin, K, 1), K

    def forward(self, x):
        alpha = F.softplus(self.conv(x).float()) + 1.0
        S = alpha.sum(1, keepdim=True)
        return alpha, alpha / S, self.K / S


class EvidentialClsHead(nn.Module):
    """Patient-level K-class evidential head: returns (alpha, prob, unc)."""
    def __init__(self, cin: int, hidden: int = CLS_HIDDEN,
                 K: int = 3, dropout: float = CLS_DROPOUT):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cin, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, K),
        )
        self.K = K

    def forward(self, v):
        alpha = F.softplus(self.mlp(v).float()) + 1.0
        S = alpha.sum(1, keepdim=True)
        return alpha, alpha / S, self.K / S


class SwinEVDNet3D(nn.Module):
    """
    3D Swin-EVDNet v2 — brain tumour segmentation + burden classification.

    Constructor (matches training):
        model = SwinEVDNet3D(num_classes=3, encoder="swin")

    Checkpoint key:
        state_dict = torch.load("best.pt", map_location="cpu",
                                weights_only=False)["model"]
        model.load_state_dict(state_dict, strict=True)

    Input:  (B, 4, D, H, W) — T1, T1ce, T2, FLAIR
    Output: dict with seg_alpha, seg_prob, seg_unc (always when need_seg=True)
            and cls_alpha, cls_prob, cls_unc (when cls_mask is provided and
            cls_head exists)
    """
    def __init__(self, num_classes: int, encoder: str):
        super().__init__()
        F_ = FEATURE_SIZE
        assert encoder == "swin", \
            "Only 'swin' encoder is supported in deployment (weights were saved with 'swin')."
        self.encoder = SwinEncoder3D(4, F_)          # 4 input modalities
        c = self.encoder.channels                    # [24, 48, 96, 192, 384]
        d8, d4, d2, d1 = DECODER_CH                 # (96, 64, 48, 32)
        self.aspp     = ASPP3D(c[3] + c[4], ASPP_CH, (1, 2, 3))
        self.up8      = UpBlock3d(ASPP_CH, c[2], d8)
        self.up4      = UpBlock3d(d8, c[1], d4)
        self.up2      = UpBlock3d(d4, c[0], d2)
        self.stem     = ConvNormAct3d(4, 16)                     # full-res skip
        self.up1      = UpBlock3d(d2, 16, d1)
        self.seg_head = EvidentialSegHead3D(d1, 2)
        self.cls_head = (EvidentialClsHead(c[4], CLS_HIDDEN, num_classes, CLS_DROPOUT)
                         if num_classes >= 2 else None)
        self.ckpt = False       # grad-checkpointing disabled for inference

    def set_grad_checkpointing(self, enable: bool) -> bool:
        self.ckpt = bool(enable)
        return self.encoder.set_grad_checkpointing(enable)

    def _run(self, mod, *xs):
        """At inference self.ckpt is False → always a direct call."""
        if self.ckpt and self.training and any(x.requires_grad for x in xs):
            import torch.utils.checkpoint as torch_ckpt
            return torch_ckpt.checkpoint(mod, *xs, use_reentrant=False)
        return mod(*xs)

    def encode(self, x):
        return self.encoder(x)         # → [s2, s4, s8, s16, s32]

    def classify(self, pooled: torch.Tensor):
        """pooled: (N_crops, 16F) → one (1, C) Dirichlet."""
        return self.cls_head(pooled.float().mean(0, keepdim=True))

    def segment(self, feats, x):
        s2, s4, s8, s16, s32 = feats
        b = self._run(self.aspp,
                      torch.cat([s16,
                                 F.interpolate(s32, size=s16.shape[-3:],
                                               mode="trilinear",
                                               align_corners=False)], 1))
        y = self._run(self.up8, b, s8)
        y = self._run(self.up4, y, s4)
        y = self._run(self.up2, y, s2)
        y = self._run(self.up1, y, self.stem(x))
        return self.seg_head(y)

    def forward(self, x: torch.Tensor,
                cls_mask: Optional[torch.Tensor] = None,
                need_seg: bool = True) -> dict:
        """
        x: (N, 4, D, H, W) — crops of ONE patient.
        cls_mask: boolean tensor selecting the mask-free classification crops.
        """
        feats = self.encode(x)
        out = {}
        if need_seg:
            out["seg_alpha"], out["seg_prob"], out["seg_unc"] = \
                self.segment(feats, x)
        if self.cls_head is not None and cls_mask is not None:
            if not bool(cls_mask.any()):
                raise ValueError("cls_mask selects no crops")
            pooled = feats[-1].float().mean(dim=(2, 3, 4))
            out["cls_alpha"], out["cls_prob"], out["cls_unc"] = \
                self.classify(pooled[cls_mask])
        return out
