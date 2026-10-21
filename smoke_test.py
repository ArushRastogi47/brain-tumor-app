"""
smoke_test.py — Verify the model loads and produces finite outputs.

Usage:
    python smoke_test.py <path/to/best_slim.pt>

Runs one forward pass with a random (1, 4, 96, 96, 96) tensor and
prints output shapes. Exits with code 0 on success.
"""
import sys
from pathlib import Path

import torch

from model import SwinEVDNet3D

if len(sys.argv) != 2:
    print("Usage: python smoke_test.py <path_to_best_slim.pt>")
    sys.exit(1)

weights_path = Path(sys.argv[1])
if not weights_path.is_file():
    print(f"ERROR: weights file not found: {weights_path}")
    sys.exit(1)

print(f"Loading weights from {weights_path} ({weights_path.stat().st_size / 1e6:.1f} MB) …")

# Constructor must match training exactly
model = SwinEVDNet3D(num_classes=3, encoder="swin")

ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

# Accept both a raw state dict and our custom swin-evdnet-v2-ckpt/2 format
if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
    state = ckpt["model"]
    print("  Checkpoint format: swin-evdnet-v2-ckpt (key='model')")
elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in list(ckpt.keys())[:5]):
    state = ckpt
    print("  Checkpoint format: plain state dict")
else:
    print(f"ERROR: unrecognised checkpoint type: {type(ckpt)}")
    sys.exit(1)

# Strip DataParallel prefix if present
n_module = sum(1 for k in state if k.startswith("module."))
if n_module > 0:
    state = {k.removeprefix("module."): v for k, v in state.items()}
    print(f"  Stripped 'module.' prefix from {n_module} keys")

model.load_state_dict(state, strict=True)
model.eval()
print("  State dict loaded (strict=True) ✓")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Forward pass with random input: shape (1, 4, 96, 96, 96)
# Use a tiny cls_mask so the classification head is also exercised
x = torch.randn(1, 4, 96, 96, 96)
cls_mask = torch.tensor([True])
print(f"  Input shape: {tuple(x.shape)}")

with torch.no_grad():
    out = model(x, cls_mask=cls_mask, need_seg=True)

print("\nOutput shapes and NaN checks:")
all_ok = True
for key, tensor in out.items():
    has_nan = bool(torch.isnan(tensor).any())
    print(f"  {key:20s} {tuple(tensor.shape)}  nan={has_nan}")
    if has_nan:
        all_ok = False

if all_ok:
    print("\nOK")
    sys.exit(0)
else:
    print("\nFAILED — NaN detected in outputs")
    sys.exit(1)
