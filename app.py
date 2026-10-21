"""
app.py — Streamlit web app for Swin-EVDNet3D brain tumor segmentation.

Research prototype. Not for medical diagnosis.

This app loads weights from a Hugging Face model repo, accepts four MRI
NIfTI uploads (T1, T1ce, T2, FLAIR), runs 3-D sliding-window inference, and
displays:
  • The mid-axial T1 slice
  • The predicted tumour mask overlay (probability > 0.5)
  • The per-voxel evidential uncertainty map
  • The patient-level tumour-burden classification (low / medium / high)

Environment variables:
  HF_TOKEN      optional; only needed if the model repo is private
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from model import SwinEVDNet3D
from preprocess import load_case

# ---------------------------------------------------------------------------
# Constants — update HF_MODEL_REPO after running Phase 6
# ---------------------------------------------------------------------------
HF_MODEL_REPO  = "arushrastogi/swin-evdnet"   # <-- update with your HF username
WEIGHTS_FILE   = "best_slim.pt"
SEG_THRESHOLD  = 0.5
ROI_SIZE       = (96, 96, 96)
SW_OVERLAP     = 0.5
SW_BATCH       = 1                             # windows per forward pass (CPU-safe)

BURDEN_THRESHOLDS_ML = [64.90, 121.13]         # from burden_thresholds.json
CLASS_NAMES          = ["low", "medium", "high"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Segmentation | Swin-EVDNet",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 8px 32px rgba(15,52,96,0.4);
}
.main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.main-header p  { font-size: 0.95rem; opacity: 0.75; margin: 0.5rem 0 0; }

.warning-box {
    background: linear-gradient(135deg, #7c2020 0%, #a83232 100%);
    border-left: 4px solid #ff6b6b;
    padding: 0.9rem 1.2rem;
    border-radius: 8px;
    color: white;
    font-weight: 600;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
}

.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.metric-card .label { font-size: 0.75rem; color: #94a3b8; font-weight: 600;
                       text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #e2e8f0;
                       margin: 0.3rem 0; }
.metric-card .sub   { font-size: 0.8rem; color: #64748b; }

.burden-badge {
    display: inline-block; padding: 0.4rem 1.2rem;
    border-radius: 999px; font-weight: 700; font-size: 1rem;
}
.burden-low    { background:#064e3b; color:#6ee7b7; }
.burden-medium { background:#78350f; color:#fde68a; }
.burden-high   { background:#7f1d1d; color:#fca5a5; }

.section-divider {
    border: none; border-top: 1px solid rgba(99,102,241,0.2);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
  <h1>🧠 Brain Tumor Segmentation</h1>
  <p>Swin-EVDNet v2 · 3D evidential segmentation & burden classification</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
  ⚠️ Research prototype. Not for medical diagnosis. Do not upload real patient data to a public server.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sliding-window inference helpers (mirrors notebook evaluate_patient)
# ---------------------------------------------------------------------------

def _window_starts(size: int, roi: int) -> list[int]:
    step = max(1, int(roi * (1.0 - SW_OVERLAP)))
    starts = list(range(0, size - roi + 1, step))
    if not starts or starts[-1] != size - roi:
        starts.append(max(0, size - roi))
    return sorted(set(starts))


def _gaussian_weights(roi: tuple[int, int, int]) -> np.ndarray:
    axes = [
        np.exp(-0.5 * ((np.arange(s) - (s - 1) / 2) / (0.125 * s)) ** 2)
        for s in roi
    ]
    w = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return np.maximum(w / w.max(), 1e-3).astype(np.float32)


@torch.inference_mode()
def sliding_window_predict(
    model: SwinEVDNet3D, image: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    3D Gaussian-weighted sliding-window inference.

    Returns (probability, uncertainty, cls_alpha) as numpy arrays.
    probability / uncertainty are 3-D (D, H, W) at the original volume size.
    """
    image = image.to(DEVICE)                              # (1, 4, D, H, W)
    vol = image[0].cpu().numpy()                          # (4, D, H, W)
    D, H, W = vol.shape[1:]
    roi = ROI_SIZE

    # Pad so each spatial dim is at least roi
    pads = [
        (max(0, r - s) // 2, max(0, r - s) - max(0, r - s) // 2)
        for s, r in zip((D, H, W), roi)
    ]
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

    prog = st.progress(0, text="Running sliding-window inference…")
    for batch_offset in range(0, len(all_starts), SW_BATCH):
        batch_starts = all_starts[batch_offset: batch_offset + SW_BATCH]
        crops = np.stack([
            padded[:, d:d+roi[0], h:h+roi[1], w:w+roi[2]]
            for d, h, w in batch_starts
        ])
        crop_tensor = torch.from_numpy(crops).to(DEVICE)
        out = model(crop_tensor)["seg_alpha"].float().cpu().numpy()  # (B,2,r,r,r)
        for idx, (d, h, w) in enumerate(batch_starts):
            sl = (slice(None), slice(d, d+roi[0]), slice(h, h+roi[1]), slice(w, w+roi[2]))
            alpha_sum[sl]  += out[idx] * gw
            weight_sum[d:d+roi[0], h:h+roi[1], w:w+roi[2]] += gw
        frac = min((batch_offset + SW_BATCH) / len(all_starts), 1.0)
        prog.progress(frac, text=f"Inference: {int(frac*100)}%")

    prog.progress(1.0, text="Done")
    alpha = alpha_sum / np.maximum(weight_sum, 1e-8)  # (2, pD, pH, pW)

    # Unpad
    crop = tuple(
        slice(before, psize - after if after else None)
        for (before, after), psize in zip(pads, (pD, pH, pW))
    )
    alpha = alpha[(slice(None),) + crop]
    strength     = alpha.sum(axis=0)
    probability  = alpha[1] / strength
    uncertainty  = 2.0 / strength

    # Classification: encode a few crops from the brain bounding box
    brain = np.any(padded != 0, axis=0)
    coords = np.argwhere(brain)
    if len(coords):
        lo, hi = coords.min(0), coords.max(0) + 1
    else:
        lo, hi = np.zeros(3, int), np.array([pD, pH, pW])
    pooled_feats = []
    for axis in range(3):
        length = int(hi[axis] - lo[axis])
        if length <= roi[axis]:
            starts_axis = [int(np.clip(lo[axis] + (length - roi[axis]) // 2,
                                       0, padded.shape[1 + axis] - roi[axis]))]
        else:
            starts_axis = sorted({
                int(np.clip(round(v), 0, padded.shape[1 + axis] - roi[axis]))
                for v in np.linspace(lo[axis], hi[axis] - roi[axis], 2)
            })
        if axis == 0:
            ax0 = starts_axis
        elif axis == 1:
            ax1 = starts_axis
        else:
            ax2 = starts_axis
    for d, h, w in itertools.product(ax0, ax1, ax2):
        crop_img = padded[:, d:d+roi[0], h:h+roi[1], w:w+roi[2]]
        feat = model.encode(
            torch.from_numpy(crop_img[None]).to(DEVICE)
        )[-1].float().mean((2, 3, 4))
        pooled_feats.append(feat)
    cls_alpha = model.classify(torch.cat(pooled_feats, 0))[0][0].cpu().numpy()

    return probability, uncertainty, cls_alpha


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model weights…")
def load_model() -> SwinEVDNet3D:
    path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=WEIGHTS_FILE,
        token=os.environ.get("HF_TOKEN"),
    )
    # Constructor must exactly match training: SwinEVDNet3D(num_classes=3, encoder="swin")
    model = SwinEVDNet3D(num_classes=3, encoder="swin")
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    # Handle both a raw state dict and our custom swin-evdnet-v2-ckpt/2 format
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in list(ckpt.keys())[:5]):
        state = ckpt
    else:
        raise RuntimeError("Unrecognised checkpoint format in best_slim.pt")
    # Strip DataParallel "module." prefix if present
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)
    return model


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold = st.slider("Segmentation threshold", 0.1, 0.9, SEG_THRESHOLD, 0.05)
    show_unc  = st.checkbox("Show uncertainty map", value=True)
    show_prob = st.checkbox("Show probability heatmap", value=False)
    st.markdown("---")
    st.markdown("**Model:** Swin-EVDNet v2 3D")
    st.markdown("**Input:** T1 · T1ce · T2 · FLAIR")
    st.markdown("**Inference:** Sliding window 96³")
    st.markdown("**Device:** `" + str(DEVICE) + "`")
    st.markdown("---")
    st.markdown(
        "Weights: [HF model repo](https://huggingface.co/"
        + HF_MODEL_REPO + ")"
    )

# ---------------------------------------------------------------------------
# Main — upload section
# ---------------------------------------------------------------------------
st.markdown("### Upload MRI Volumes")
st.caption("Upload all four co-registered NIfTI volumes (.nii or .nii.gz).")

col1, col2, col3, col4 = st.columns(4)
with col1:
    t1    = st.file_uploader("T1",    type=["nii", "gz"], key="t1")
with col2:
    t1ce  = st.file_uploader("T1ce",  type=["nii", "gz"], key="t1ce")
with col3:
    t2    = st.file_uploader("T2",    type=["nii", "gz"], key="t2")
with col4:
    flair = st.file_uploader("FLAIR", type=["nii", "gz"], key="flair")

all_uploaded = all([t1, t1ce, t2, flair])

if all_uploaded:
    st.success("✓ All four modalities uploaded. Click **Run segmentation** to proceed.")
else:
    missing = [m for m, f in [("T1", t1), ("T1ce", t1ce), ("T2", t2), ("FLAIR", flair)] if not f]
    if missing:
        st.info(f"Waiting for: {', '.join(missing)}")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
if all_uploaded and st.button("🔬 Run segmentation", type="primary", use_container_width=True):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    with st.spinner("Preprocessing volumes…"):
        try:
            x, display_img = load_case(t1, t1ce, t2, flair)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

    probability, uncertainty, cls_alpha = sliding_window_predict(model, x)

    # --- Metrics ---
    mask = probability > threshold
    st.markdown("### Results")
    mc1, mc2, mc3, mc4 = st.columns(4)

    tumor_voxels = int(mask.sum())
    # approximate mm³ from voxel count (real mm³ needs the affine — show voxels)
    cls_strength = float(cls_alpha.sum())
    cls_probs    = cls_alpha / cls_strength
    burden_idx   = int(np.argmax(cls_alpha))
    burden_name  = CLASS_NAMES[burden_idx]
    seg_unc_mean = float(uncertainty[mask].mean()) if tumor_voxels > 0 else float("nan")

    with mc1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Tumor Voxels</div>
            <div class="value">{tumor_voxels:,}</div>
            <div class="sub">threshold={threshold:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with mc2:
        badge_cls = f"burden-{burden_name}"
        st.markdown(f"""<div class="metric-card">
            <div class="label">Burden Class</div>
            <div class="value">
              <span class="burden-badge {badge_cls}">{burden_name.upper()}</span>
            </div>
            <div class="sub">model classification</div>
        </div>""", unsafe_allow_html=True)

    with mc3:
        conf = float(cls_probs[burden_idx]) * 100
        st.markdown(f"""<div class="metric-card">
            <div class="label">Cls Confidence</div>
            <div class="value">{conf:.1f}%</div>
            <div class="sub">Dirichlet probability</div>
        </div>""", unsafe_allow_html=True)

    with mc4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Mean Uncertainty</div>
            <div class="value">{seg_unc_mean:.3f}</div>
            <div class="sub">within tumor mask</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # --- Visualisation ---
    st.markdown("### Visualisation")

    # Pick mid-axial slice through the predicted mask centre if possible
    if tumor_voxels > 0:
        coords  = np.argwhere(mask)
        mid_d   = int(coords[:, 0].mean())
    else:
        mid_d = display_img.shape[0] // 2 if display_img.ndim == 2 else probability.shape[0] // 2

    disp  = display_img if display_img.ndim == 2 else None
    if disp is None:
        disp = np.zeros((probability.shape[1], probability.shape[2]))

    mask_slice = mask[mid_d]
    prob_slice = probability[mid_d]
    unc_slice  = uncertainty[mid_d]

    n_plots = 2 + int(show_unc) + int(show_prob)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5),
                              facecolor="#0f172a")
    plt.subplots_adjust(left=0.02, right=0.98, wspace=0.06)

    def _setup_ax(ax, title):
        ax.set_facecolor("#0f172a")
        ax.axis("off")
        ax.set_title(title, color="#e2e8f0", fontsize=10, pad=6, fontweight="bold")

    ax_idx = 0
    _setup_ax(axes[ax_idx], "MRI — T1 (mid slice)")
    axes[ax_idx].imshow(disp, cmap="gray", interpolation="bilinear")
    ax_idx += 1

    _setup_ax(axes[ax_idx], "Predicted tumour mask")
    axes[ax_idx].imshow(disp, cmap="gray", interpolation="bilinear")
    overlay = np.zeros((*mask_slice.shape, 4))
    overlay[mask_slice] = [1.0, 0.2, 0.2, 0.55]
    axes[ax_idx].imshow(overlay)
    ax_idx += 1

    if show_prob:
        _setup_ax(axes[ax_idx], "Tumour probability")
        axes[ax_idx].imshow(disp, cmap="gray", interpolation="bilinear", alpha=0.5)
        im_p = axes[ax_idx].imshow(prob_slice, cmap="plasma", vmin=0, vmax=1, alpha=0.7)
        plt.colorbar(im_p, ax=axes[ax_idx], fraction=0.046, pad=0.04).ax.tick_params(colors="white")
        ax_idx += 1

    if show_unc:
        _setup_ax(axes[ax_idx], "Evidential uncertainty")
        im_u = axes[ax_idx].imshow(unc_slice, cmap="viridis", interpolation="bilinear")
        plt.colorbar(im_u, ax=axes[ax_idx], fraction=0.046, pad=0.04).ax.tick_params(colors="white")
        ax_idx += 1

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # --- Classification bar ---
    st.markdown("### Burden classification probabilities")
    bar_cols = st.columns(3)
    colours  = ["#10b981", "#f59e0b", "#ef4444"]
    for i, (cname, cprob, colour) in enumerate(
        zip(CLASS_NAMES, cls_probs.tolist(), colours)
    ):
        with bar_cols[i]:
            st.markdown(f"**{cname.upper()}**")
            st.progress(float(cprob))
            st.caption(f"{cprob*100:.1f}%")

    # Final medical disclaimer
    st.markdown("---")
    st.caption(
        "**Research prototype. Not for medical diagnosis.** "
        "This model was trained on public BraTS data for 25 epochs. "
        "Tumour-burden categories are volume tertiles from the training split "
        "and are not equivalent to clinical HGG/LGG grading."
    )
