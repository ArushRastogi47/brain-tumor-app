"""
app.py — Gradio web app for Swin-EVDNet3D brain tumour segmentation.

Research prototype. Not for medical diagnosis.

Accepts four MRI NIfTI uploads (T1, T1ce, T2, FLAIR) and returns:
  • Tumour probability map overlay (mid-axial slice)
  • Evidential uncertainty map
  • Tumour-burden classification (low / medium / high) with probabilities
"""

from __future__ import annotations

import itertools
import os
import tempfile

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from model import SwinEVDNet3D
from preprocess import load_patient_from_paths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_MODEL_REPO        = "aryan09103/swin-evdnet"
WEIGHTS_FILE         = "best_slim.pt"
SEG_THRESHOLD        = 0.5
ROI_SIZE             = (96, 96, 96)
SW_OVERLAP           = 0.5
SW_BATCH             = 1
BURDEN_THRESHOLDS_ML = [64.90, 121.13]
CLASS_NAMES          = ["low", "medium", "high"]
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model loader (loaded once at startup)
# ---------------------------------------------------------------------------
_MODEL = None

def get_model() -> SwinEVDNet3D:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    local_path = os.environ.get("MODEL_WEIGHTS_PATH")
    if local_path and os.path.isfile(local_path):
        path = local_path
        print(f"[model] Using local weights: {path}")
    else:
        print(f"[model] Downloading from HF: {HF_MODEL_REPO}/{WEIGHTS_FILE}")
        path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=WEIGHTS_FILE,
            token=os.environ.get("HF_TOKEN"),
        )

    model = SwinEVDNet3D(num_classes=3, encoder="swin")
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in list(ckpt.keys())[:5]):
        state = ckpt
    else:
        raise RuntimeError("Unrecognised checkpoint format.")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)
    print(f"[model] Loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params) on {DEVICE}")
    _MODEL = model
    return model


# ---------------------------------------------------------------------------
# Sliding-window helpers
# ---------------------------------------------------------------------------
def _window_starts(size: int, roi: int) -> list[int]:
    step = max(1, int(roi * (1.0 - SW_OVERLAP)))
    starts = list(range(0, size - roi + 1, step))
    if not starts or starts[-1] != size - roi:
        starts.append(max(0, size - roi))
    return sorted(set(starts))


def _gaussian_weights(roi) -> np.ndarray:
    axes = [np.exp(-0.5 * ((np.arange(s) - (s-1)/2) / (0.125*s))**2) for s in roi]
    w = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return np.maximum(w / w.max(), 1e-3).astype(np.float32)


@torch.inference_mode()
def sliding_window_predict(model, image: torch.Tensor):
    """image: (1, 4, D, H, W) float32 tensor. Returns (prob, unc, cls_alpha)."""
    image = image.to(DEVICE)
    vol   = image[0].cpu().numpy()                  # (4, D, H, W)
    D, H, W = vol.shape[1:]
    roi = ROI_SIZE

    pads = [(max(0, r-s)//2, max(0, r-s) - max(0, r-s)//2)
            for s, r in zip((D, H, W), roi)]
    padded = np.pad(vol, [(0, 0), *pads], mode="constant")
    pD, pH, pW = padded.shape[1:]

    alpha_sum  = np.zeros((2, pD, pH, pW), dtype=np.float64)
    weight_sum = np.zeros((pD, pH, pW),    dtype=np.float64)
    gw = _gaussian_weights(roi)

    all_starts = list(itertools.product(
        _window_starts(pD, roi[0]),
        _window_starts(pH, roi[1]),
        _window_starts(pW, roi[2]),
    ))
    for offset in range(0, len(all_starts), SW_BATCH):
        batch_s = all_starts[offset: offset + SW_BATCH]
        crops   = np.stack([padded[:, d:d+roi[0], h:h+roi[1], w:w+roi[2]]
                            for d, h, w in batch_s])
        out = model(torch.from_numpy(crops).to(DEVICE))["seg_alpha"].float().cpu().numpy()
        for idx, (d, h, w) in enumerate(batch_s):
            sl = (slice(None), slice(d,d+roi[0]), slice(h,h+roi[1]), slice(w,w+roi[2]))
            alpha_sum[sl]                           += out[idx] * gw
            weight_sum[d:d+roi[0], h:h+roi[1], w:w+roi[2]] += gw

    alpha = alpha_sum / np.maximum(weight_sum, 1e-8)
    crop  = tuple(slice(b, psize - a if a else None)
                  for (b, a), psize in zip(pads, (pD, pH, pW)))
    alpha = alpha[(slice(None),) + crop]
    prob  = alpha[1] / alpha.sum(0)
    unc   = 2.0 / alpha.sum(0)

    # Classification crops from brain bounding box
    brain  = np.any(padded != 0, axis=0)
    coords = np.argwhere(brain)
    lo = coords.min(0) if len(coords) else np.zeros(3, int)
    hi = coords.max(0)+1 if len(coords) else np.array([pD, pH, pW])
    pooled_feats = []
    axes_list = []
    for ax in range(3):
        length = int(hi[ax] - lo[ax])
        if length <= roi[ax]:
            axes_list.append([int(np.clip(lo[ax] + (length-roi[ax])//2,
                                          0, padded.shape[1+ax]-roi[ax]))])
        else:
            axes_list.append(sorted({int(np.clip(round(v), 0,
                                                  padded.shape[1+ax]-roi[ax]))
                                      for v in np.linspace(lo[ax], hi[ax]-roi[ax], 2)}))
    for d, h, w in itertools.product(*axes_list):
        ci = padded[:, d:d+roi[0], h:h+roi[1], w:w+roi[2]]
        f  = model.encode(torch.from_numpy(ci[None]).to(DEVICE))[-1].float().mean((2,3,4))
        pooled_feats.append(f)
    cls_alpha = model.classify(torch.cat(pooled_feats, 0))[0][0].cpu().numpy()
    return prob, unc, cls_alpha


# ---------------------------------------------------------------------------
# Inference function called by Gradio
# ---------------------------------------------------------------------------
def run_segmentation(t1_file, t1ce_file, t2_file, flair_file, threshold):
    if any(f is None for f in [t1_file, t1ce_file, t2_file, flair_file]):
        error_html = """
<div style="background:rgba(239, 68, 68, 0.12); border:1px solid rgba(239, 68, 68, 0.35); border-left:4px solid #ef4444; border-radius:10px; padding:1rem 1.25rem; color:#fca5a5; margin:0.5rem 0;">
  <div style="font-weight:700; font-size:0.95rem; display:flex; align-items:center; gap:0.4rem; margin-bottom:0.25rem;">
    <span>⚠️</span> Missing Input Files
  </div>
  <p style="font-size:0.85rem; margin:0; line-height:1.4;">
    Please upload all four required MRI modalities (<strong>T1</strong>, <strong>T1ce</strong>, <strong>T2</strong>, and <strong>FLAIR</strong>) in .nii or .nii.gz format before running segmentation.
  </p>
</div>
"""
        return None, error_html

    from pathlib import Path
    paths = {
        "t1":    Path(t1_file),
        "t1ce":  Path(t1ce_file),
        "t2":    Path(t2_file),
        "flair": Path(flair_file),
    }

    try:
        model = get_model()
        x, reference = load_patient_from_paths(paths)
        prob, unc, cls_alpha = sliding_window_predict(model, x)
    except Exception as e:
        error_html = f"""
<div style="background:rgba(239, 68, 68, 0.12); border:1px solid rgba(239, 68, 68, 0.35); border-left:4px solid #ef4444; border-radius:10px; padding:1rem 1.25rem; color:#fca5a5; margin:0.5rem 0;">
  <div style="font-weight:700; font-size:0.95rem; display:flex; align-items:center; gap:0.4rem; margin-bottom:0.25rem;">
    <span>❌</span> Processing Error
  </div>
  <p style="font-size:0.85rem; margin:0; line-height:1.4;">{str(e)}</p>
</div>
"""
        return None, error_html

    mask = prob > threshold

    # --- Metrics ---
    tumor_voxels = int(mask.sum())
    voxel_ml = abs(float(np.linalg.det(reference.affine[:3, :3]))) / 1000.0
    volume_ml = tumor_voxels * voxel_ml
    cls_strength = float(cls_alpha.sum())
    cls_probs    = cls_alpha / cls_strength
    burden_idx   = int(np.argmax(cls_alpha))
    burden_name  = CLASS_NAMES[burden_idx]
    seg_derived_idx = int(np.digitize(volume_ml, BURDEN_THRESHOLDS_ML, right=False))
    seg_derived_name = CLASS_NAMES[seg_derived_idx]
    seg_unc_mean = float(unc[mask].mean()) if tumor_voxels > 0 else float("nan")

    # --- Mid-axial slice (through mask centre if tumour exists) ---
    if tumor_voxels > 0:
        mid_d = int(np.argwhere(mask)[:, 0].mean())
    else:
        mid_d = prob.shape[0] // 2

    # T1 channel for background
    x_np  = x[0, 0].numpy()            # (D, H, W)
    bg    = x_np[mid_d]
    u_sl  = unc[mid_d]
    m_sl  = mask[mid_d]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor="#0b0f19")
    plt.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.08, wspace=0.12)

    def _ax(ax, title, subtitle=""):
        ax.set_facecolor("#0b0f19")
        ax.axis("off")
        ax.set_title(title, color="#f1f5f9", fontsize=11, fontweight="bold", pad=8)
        if subtitle:
            ax.text(0.5, -0.06, subtitle, transform=ax.transAxes,
                    color="#94a3b8", fontsize=8.5, ha="center")

    _ax(axes[0], "1. Anatomical T1", f"Mid-axial Slice Z={mid_d}")
    axes[0].imshow(bg, cmap="gray")

    _ax(axes[1], "2. Predicted Tumor Mask", f"Thresh = {threshold:.2f} · Vol = {volume_ml:.1f} mL")
    axes[1].imshow(bg, cmap="gray")
    if m_sl.any():
        overlay = np.zeros((*m_sl.shape, 4))
        overlay[m_sl] = [0.96, 0.25, 0.37, 0.65]  # Crisp Rose/Crimson
        axes[1].imshow(overlay)
        axes[1].contour(m_sl, colors=["#f43f5e"], linewidths=1.2)

    _ax(axes[2], "3. Evidential Uncertainty", "Epistemic Uncertainty (2/S)")
    im = axes[2].imshow(u_sl, cmap="viridis")
    cb = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#94a3b8", labelsize=8)
    cb.outline.set_edgecolor("#334155")

    # Save to temp
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, bbox_inches="tight", dpi=140, facecolor="#0b0f19")
    plt.close(fig)

    # --- Result report HTML ---
    unc_str = f"{seg_unc_mean:.4f}" if not np.isnan(seg_unc_mean) else "N/A"
    low_pct = float(cls_probs[0] * 100)
    med_pct = float(cls_probs[1] * 100)
    high_pct = float(cls_probs[2] * 100)

    report = f"""
<div class="results-container">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:0.5rem">
    <div style="font-size:1.1rem; font-weight:700; color:#f8fafc; display:flex; align-items:center; gap:0.5rem">
      <span>📊</span> Clinical Inference Telemetry
    </div>
    <span class="burden-pill {burden_name}">
      {burden_name.upper()} BURDEN
    </span>
  </div>

  <div class="kpi-grid">
    <div class="kpi-tile">
      <div class="kpi-title">Tumor Volume</div>
      <div class="kpi-num">{volume_ml:.1f} <span style="font-size:0.85rem; font-weight:500; color:#94a3b8">mL</span></div>
      <div class="kpi-sub">{tumor_voxels:,} voxels</div>
    </div>

    <div class="kpi-tile">
      <div class="kpi-title">Model Burden</div>
      <div class="kpi-num" style="font-size:1.25rem"><span class="burden-pill {burden_name}">{burden_name.upper()}</span></div>
      <div class="kpi-sub">Confidence: {cls_probs[burden_idx]*100:.1f}%</div>
    </div>

    <div class="kpi-tile">
      <div class="kpi-title">Seg-Derived</div>
      <div class="kpi-num" style="font-size:1.25rem"><span class="burden-pill {seg_derived_name}">{seg_derived_name.upper()}</span></div>
      <div class="kpi-sub">Volume cutoffs</div>
    </div>

    <div class="kpi-tile">
      <div class="kpi-title">Mean Epistemic Unc.</div>
      <div class="kpi-num">{unc_str}</div>
      <div class="kpi-sub">Evidential (2/S)</div>
    </div>
  </div>

  <div class="prob-card">
    <div style="font-size:0.8rem; font-weight:700; color:#cbd5e1; margin-bottom:0.6rem; text-transform:uppercase; letter-spacing:0.04em">
      Burden Probability Distribution (Dirichlet α)
    </div>
    <div class="prob-item">
      <div class="prob-header">
        <span>Low Burden (&lt;64.9 mL)</span>
        <span style="font-family:var(--font-mono)">{low_pct:.1f}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-bar bar-low" style="width: {low_pct:.1f}%"></div>
      </div>
    </div>
    <div class="prob-item">
      <div class="prob-header">
        <span>Medium Burden (64.9–121.1 mL)</span>
        <span style="font-family:var(--font-mono)">{med_pct:.1f}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-bar bar-medium" style="width: {med_pct:.1f}%"></div>
      </div>
    </div>
    <div class="prob-item">
      <div class="prob-header">
        <span>High Burden (&gt;121.1 mL)</span>
        <span style="font-family:var(--font-mono)">{high_pct:.1f}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-bar bar-high" style="width: {high_pct:.1f}%"></div>
      </div>
    </div>
  </div>
</div>
"""
    return tmp.name, report


# ---------------------------------------------------------------------------
# Gradio UI Design & Layout
# ---------------------------------------------------------------------------
HEADER_HTML = """
<div class="app-header">
  <div class="badge-row">
    <span class="header-badge primary">🧠 Swin-EVDNet v2 3D</span>
    <span class="header-badge success">● Engine Ready</span>
    <span class="header-badge">BraTS 2023 Glioma</span>
    <span class="header-badge">4.17M Parameters</span>
  </div>
  <h1 class="app-title">Brain Tumor Segmentation & Evidential Analysis</h1>
  <p class="app-subtitle">
    3D Swin Transformer with Evidential Deep Learning for multi-parametric MRI segmentation, volumetric quantification, and epistemic uncertainty mapping.
  </p>
  <div class="disclaimer-box">
    <span style="font-size:1.25rem;">⚠️</span>
    <div>
      <strong>Research Prototype Only:</strong> This application is intended strictly for research and evaluation purposes and is not certified for clinical diagnostic decision-making.
    </div>
  </div>
</div>
"""

EMPTY_STATE_HTML = """
<div class="empty-state">
  <div class="empty-state-icon">🧠</div>
  <h3>Model Ready for Inference</h3>
  <p>Upload the four required MRI scans (<strong>T1</strong>, <strong>T1ce</strong>, <strong>T2</strong>, and <strong>FLAIR</strong>) in the left panel and click <strong>"Run Segmentation"</strong> to visualize 3D tumor contours and uncertainty maps.</p>
</div>
"""

FOOTER_HTML = """
<div class="app-footer">
  <div>
    <strong>Architecture:</strong> Swin-EVDNet v2 3D · <strong>Resolution:</strong> 1mm³ isotropic · <strong>Task:</strong> Whole Tumor (WT)
  </div>
  <div>
    <strong>Weights:</strong> <a href="https://huggingface.co/aryan09103/swin-evdnet" target="_blank">aryan09103/swin-evdnet</a>
  </div>
</div>
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#0b0f19",
    body_background_fill_dark="#0b0f19",
    block_background_fill="rgba(17, 24, 39, 0.75)",
    block_background_fill_dark="rgba(17, 24, 39, 0.75)",
    block_border_color="rgba(255, 255, 255, 0.08)",
    block_border_color_dark="rgba(255, 255, 255, 0.08)",
    input_background_fill="rgba(15, 23, 42, 0.9)",
    input_background_fill_dark="rgba(15, 23, 42, 0.9)",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

body, .gradio-container {
  font-family: var(--font-sans) !important;
  background-color: #0b0f19 !important;
  color: #f1f5f9 !important;
}

.gradio-container {
  background: 
    radial-gradient(circle at 12% 10%, rgba(99, 102, 241, 0.12) 0%, transparent 35%),
    radial-gradient(circle at 88% 18%, rgba(139, 92, 246, 0.10) 0%, transparent 40%),
    radial-gradient(circle at 50% 90%, rgba(6, 182, 212, 0.06) 0%, transparent 50%),
    #0b0f19 !important;
  max-width: 1380px !important;
  margin: 0 auto !important;
  padding: 1.25rem 1rem !important;
}

.app-header {
  text-align: center;
  padding: 1rem 1rem 1.75rem 1rem;
  margin-bottom: 1.25rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.07);
}

.app-title {
  font-size: 2.15rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.03em !important;
  margin: 0.4rem 0 !important;
  background: linear-gradient(135deg, #ffffff 20%, #e2e8f0 60%, #94a3b8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.app-subtitle {
  color: #94a3b8;
  font-size: 0.98rem;
  max-width: 720px;
  margin: 0 auto 1rem auto;
  line-height: 1.5;
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  justify-content: center;
  align-items: center;
  margin-bottom: 0.75rem;
}

.header-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.25rem 0.7rem;
  font-size: 0.74rem;
  font-weight: 600;
  border-radius: 9999px;
  background: rgba(30, 41, 59, 0.7);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #cbd5e1;
}

.header-badge.primary {
  background: rgba(99, 102, 241, 0.15);
  border-color: rgba(99, 102, 241, 0.35);
  color: #a5b4fc;
}

.header-badge.success {
  background: rgba(16, 185, 129, 0.15);
  border-color: rgba(16, 185, 129, 0.35);
  color: #6ee7b7;
}

.disclaimer-box {
  background: rgba(239, 68, 68, 0.09);
  border: 1px solid rgba(239, 68, 68, 0.25);
  border-left: 4px solid #ef4444;
  border-radius: 8px;
  padding: 0.6rem 0.9rem;
  max-width: 680px;
  margin: 0 auto;
  text-align: left;
  font-size: 0.83rem;
  color: #fca5a5;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 1rem;
  font-weight: 700;
  color: #e2e8f0;
  margin-bottom: 0.75rem;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.btn-infer {
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  font-weight: 700 !important;
  font-size: 1.02rem !important;
  letter-spacing: 0.015em !important;
  padding: 0.85rem 1.5rem !important;
  border-radius: 10px !important;
  box-shadow: 0 4px 18px rgba(99, 102, 241, 0.4) !important;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
  cursor: pointer !important;
  margin-top: 0.6rem !important;
}
.btn-infer:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 26px rgba(124, 58, 237, 0.6) !important;
}

.file-upload-box {
  border: 1px dashed rgba(255, 255, 255, 0.14) !important;
  border-radius: 9px !important;
  transition: all 0.2s ease !important;
}
.file-upload-box:hover {
  border-color: rgba(99, 102, 241, 0.45) !important;
  background: rgba(99, 102, 241, 0.03) !important;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 0.7rem;
  margin: 0.85rem 0;
}

.kpi-tile {
  background: rgba(15, 23, 42, 0.85);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 9px;
  padding: 0.8rem 0.9rem;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-tile:hover {
  border-color: rgba(99, 102, 241, 0.4);
  transform: translateY(-2px);
}

.kpi-title {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  color: #94a3b8;
  letter-spacing: 0.04em;
  margin-bottom: 0.3rem;
}

.kpi-num {
  font-size: 1.45rem;
  font-weight: 800;
  color: #f8fafc;
  line-height: 1.1;
  font-family: var(--font-sans);
}

.kpi-sub {
  font-size: 0.72rem;
  color: #64748b;
  margin-top: 0.3rem;
}

.burden-pill {
  display: inline-block;
  padding: 0.22rem 0.6rem;
  font-size: 0.8rem;
  font-weight: 800;
  border-radius: 6px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.burden-pill.low { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
.burden-pill.medium { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
.burden-pill.high { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }

.prob-card {
  background: rgba(15, 23, 42, 0.7);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 9px;
  padding: 0.85rem 0.95rem;
  margin-top: 0.75rem;
}

.prob-item {
  margin-bottom: 0.6rem;
}
.prob-item:last-child {
  margin-bottom: 0;
}

.prob-header {
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
  font-weight: 600;
  color: #cbd5e1;
  margin-bottom: 0.25rem;
}

.prob-track {
  width: 100%;
  height: 8px;
  background: rgba(30, 41, 59, 0.8);
  border-radius: 9999px;
  overflow: hidden;
}

.prob-bar {
  height: 100%;
  border-radius: 9999px;
  transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.bar-low { background: linear-gradient(90deg, #10b981, #34d399); }
.bar-medium { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.bar-high { background: linear-gradient(90deg, #ef4444, #f87171); }

.empty-state {
  text-align: center;
  padding: 2.8rem 1.25rem;
  background: rgba(15, 23, 42, 0.45);
  border: 1px dashed rgba(255, 255, 255, 0.1);
  border-radius: 11px;
  color: #64748b;
}

.empty-state-icon {
  font-size: 2.5rem;
  margin-bottom: 0.6rem;
  opacity: 0.85;
}

.empty-state h3 {
  color: #cbd5e1;
  font-size: 1.1rem;
  font-weight: 700;
  margin-bottom: 0.35rem;
}

.empty-state p {
  font-size: 0.85rem;
  max-width: 400px;
  margin: 0 auto;
  line-height: 1.5;
}

.app-footer {
  margin-top: 1.75rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  font-size: 0.8rem;
  color: #64748b;
  gap: 0.75rem;
}

.app-footer a {
  color: #818cf8;
  text-decoration: none;
}
.app-footer a:hover {
  text-decoration: underline;
}

footer { display: none !important; }
"""

with gr.Blocks(title="Brain Tumor Segmentation | Swin-EVDNet") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):
        # Left column: Controls & Uploads
        with gr.Column(scale=5):
            gr.HTML("""
            <div class="section-title">
              <span>📥</span> Modality Uploads (4 Scans Required)
            </div>
            """)
            with gr.Row():
                t1_up = gr.File(
                    label="T1 (Native)",
                    file_types=[".nii", ".gz"],
                    elem_classes=["file-upload-box"],
                )
                t1ce_up = gr.File(
                    label="T1ce (Contrast)",
                    file_types=[".nii", ".gz"],
                    elem_classes=["file-upload-box"],
                )
            with gr.Row():
                t2_up = gr.File(
                    label="T2 Weighted",
                    file_types=[".nii", ".gz"],
                    elem_classes=["file-upload-box"],
                )
                flair_up = gr.File(
                    label="FLAIR Inversion",
                    file_types=[".nii", ".gz"],
                    elem_classes=["file-upload-box"],
                )

            gr.HTML("""
            <div class="section-title" style="margin-top:1.25rem;">
              <span>⚙️</span> Segmentation Sensitivity
            </div>
            """)
            threshold_sl = gr.Slider(
                0.1, 0.9, value=SEG_THRESHOLD, step=0.05,
                label="Probability Cutoff (Threshold)",
                info="Default 0.50. Higher values yield high specificity; lower values capture diffuse edema margins.",
            )
            run_btn = gr.Button("🔬 Run Segmentation & Uncertainty Profiling", variant="primary", elem_classes=["btn-infer"])

        # Right column: Outputs & Telemetry
        with gr.Column(scale=7):
            gr.HTML("""
            <div class="section-title">
              <span>🖥️</span> Inference Visualisation & Clinical Report
            </div>
            """)
            out_img = gr.Image(label="Multi-slice Segmentation & Epistemic Uncertainty", type="filepath")
            out_text = gr.HTML(EMPTY_STATE_HTML, label="Clinical Report")

    run_btn.click(
        fn=run_segmentation,
        inputs=[t1_up, t1ce_up, t2_up, flair_up, threshold_sl],
        outputs=[out_img, out_text],
    )

    gr.HTML(FOOTER_HTML)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS, server_name="0.0.0.0", server_port=7860)
