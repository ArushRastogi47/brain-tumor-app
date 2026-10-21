"""
preprocess.py — inference-time preprocessing for SwinEVDNet3D.

Mirrors the validation pipeline from swin_evdnet_v2_native_3d.ipynb exactly:
  1. Load each modality as a 3D NIfTI volume.
  2. Resample non-T1 modalities to the T1 grid if grids differ.
  3. Z-score normalise over non-zero, finite voxels per channel (same as
     zscore_nonzero() used during training).
  4. Stack channels in the order: T1, T1ce, T2, FLAIR  (MODALITIES tuple).
  5. Return a float32 tensor of shape (1, 4, D, H, W) ready for sliding-window
     inference, plus the reference NIfTI image for result saving.

No augmentation is applied (training used none; validation used none).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import nibabel.processing as nib_processing
import numpy as np
import torch

MODALITIES = ("t1", "t1ce", "t2", "flair")


# ---------------------------------------------------------------------------
# Core preprocessing helpers
# ---------------------------------------------------------------------------

def zscore_nonzero(volume: np.ndarray) -> np.ndarray:
    """Z-score over finite non-zero voxels; zero-pad elsewhere.  float32 out."""
    out = np.zeros(volume.shape, dtype=np.float32)
    finite = np.isfinite(volume)
    mask = finite & (volume != 0)
    if int(mask.sum()) < 10:
        return out
    values = volume[mask].astype(np.float64)
    mean, std = float(values.mean()), float(values.std())
    if not np.isfinite(std) or std < 1e-6:
        return out
    out[mask] = ((values - mean) / std).astype(np.float32)
    return out


def _as_3d(image: nib.Nifti1Image) -> nib.Nifti1Image:
    """Drop trailing singleton dimensions so the volume is exactly 3-D."""
    if len(image.shape) < 3:
        raise ValueError("NIfTI image has fewer than 3 dimensions.")
    if len(image.shape) == 3:
        return image
    if any(s != 1 for s in image.shape[3:]):
        raise ValueError(
            f"NIfTI image is {len(image.shape)}-D with non-singleton trailing "
            f"dims {image.shape[3:]}; cannot reduce to 3-D."
        )
    data = np.asarray(image.dataobj)[..., *([0] * len(image.shape[3:]))]
    return nib.Nifti1Image(data, image.affine, image.header)


def _same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image) -> bool:
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=1e-4)


def load_patient_from_paths(paths: dict[str, Path]) -> tuple[torch.Tensor, nib.Nifti1Image]:
    """
    Load four modalities from file paths, preprocess, and return:
      tensor : float32 (1, 4, D, H, W)
      reference : T1 NIfTI image (used for output saving in smoke_test)
    """
    reference = _as_3d(nib.load(str(paths["t1"])))
    ref_shape = tuple(int(s) for s in reference.shape[:3])
    channels = []
    for mod in MODALITIES:
        img = _as_3d(nib.load(str(paths[mod])))
        if not _same_grid(img, reference):
            img = nib_processing.resample_from_to(
                img, (ref_shape, reference.affine),
                order=1, mode="constant", cval=0.0,
            )
        channels.append(zscore_nonzero(img.get_fdata(dtype=np.float32)))
    stacked = np.stack(channels, axis=0)  # (4, D, H, W)
    if not np.isfinite(stacked).all():
        raise ValueError("Non-finite values after normalisation.")
    tensor = torch.from_numpy(stacked).unsqueeze(0)  # (1, 4, D, H, W)
    return tensor, reference


# ---------------------------------------------------------------------------
# Streamlit upload interface
# ---------------------------------------------------------------------------

def read_nifti(uploaded) -> nib.Nifti1Image:
    """Save a Streamlit UploadedFile to a temp file and load with nibabel."""
    suffix = ".nii.gz" if uploaded.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(uploaded.getvalue())
        tmp_path = f.name
    return nib.load(tmp_path)


def load_case(
    t1, t1ce, t2, flair
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Takes four Streamlit UploadedFile objects (T1, T1ce, T2, FLAIR).
    Returns:
      x           : float32 tensor (1, 4, D, H, W)
      display_img : 2D float32 numpy array (mid-axial slice of T1 for display)
    """
    imgs_nib = {
        "t1":    _as_3d(read_nifti(t1)),
        "t1ce":  _as_3d(read_nifti(t1ce)),
        "t2":    _as_3d(read_nifti(t2)),
        "flair": _as_3d(read_nifti(flair)),
    }
    reference = imgs_nib["t1"]
    ref_shape = tuple(int(s) for s in reference.shape[:3])
    channels = []
    for mod in MODALITIES:
        img = imgs_nib[mod]
        if not _same_grid(img, reference):
            img = nib_processing.resample_from_to(
                img, (ref_shape, reference.affine),
                order=1, mode="constant", cval=0.0,
            )
        channels.append(zscore_nonzero(img.get_fdata(dtype=np.float32)))
    stacked = np.stack(channels, axis=0)  # (4, D, H, W)
    if not np.isfinite(stacked).all():
        raise ValueError("Non-finite values detected after normalisation.")
    # Mid-axial slice of the T1 channel for display
    mid = stacked.shape[1] // 2
    display_img = stacked[0, mid, :, :]
    tensor = torch.from_numpy(stacked).unsqueeze(0)  # (1, 4, D, H, W)
    return tensor, display_img
