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
        return None, "❌ Please upload all four MRI modalities."

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
        return None, f"❌ Error: {e}"

    mask = prob > threshold

    # --- Metrics ---
    tumor_voxels = int(mask.sum())
    voxel_ml = abs(float(np.linalg.det(reference.affine[:3, :3]))) / 1000.0
    volume_ml = tumor_voxels * voxel_ml
    cls_strength = float(cls_alpha.sum())
    cls_probs    = cls_alpha / cls_strength
    burden_idx   = int(np.argmax(cls_alpha))
    burden_name  = CLASS_NAMES[burden_idx]
    seg_unc_mean = float(unc[mask].mean()) if tumor_voxels > 0 else float("nan")

    # --- Mid-axial slice (through mask centre if tumour exists) ---
    if tumor_voxels > 0:
        mid_d = int(np.argwhere(mask)[:, 0].mean())
    else:
        mid_d = prob.shape[0] // 2

    # T1 channel for background
    x_np  = x[0, 0].numpy()            # (D, H, W)
    bg    = x_np[mid_d]
    p_sl  = prob[mid_d]
    u_sl  = unc[mid_d]
    m_sl  = mask[mid_d]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#0f172a")
    plt.subplots_adjust(left=0.02, right=0.98, wspace=0.08)

    def _ax(ax, title):
        ax.set_facecolor("#0f172a")
        ax.axis("off")
        ax.set_title(title, color="#e2e8f0", fontsize=10, fontweight="bold", pad=5)

    _ax(axes[0], "T1 — mid-axial slice")
    axes[0].imshow(bg, cmap="gray")

    _ax(axes[1], f"Tumour mask  (thresh={threshold:.2f})")
    axes[0].imshow(bg, cmap="gray")
    axes[1].imshow(bg, cmap="gray")
    overlay = np.zeros((*m_sl.shape, 4))
    overlay[m_sl] = [1.0, 0.2, 0.2, 0.55]
    axes[1].imshow(overlay)

    _ax(axes[2], "Evidential uncertainty")
    im = axes[2].imshow(u_sl, cmap="viridis")
    cb = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#94a3b8", labelsize=7)

    # Save to temp
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, bbox_inches="tight", dpi=120, facecolor="#0f172a")
    plt.close(fig)

    # --- Result text ---
    cls_bar = "  ".join(
        f"{n.upper()}: {p*100:.1f}%"
        for n, p in zip(CLASS_NAMES, cls_probs.tolist())
    )
    report = (
        f"## Results\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Tumour voxels | {tumor_voxels:,} |\n"
        f"| Tumour volume | {volume_ml:.1f} mL |\n"
        f"| Burden class (seg-derived) | **{CLASS_NAMES[int(np.digitize(volume_ml, BURDEN_THRESHOLDS_ML, right=False))].upper()}** |\n"
        f"| Burden class (model) | **{burden_name.upper()}** |\n"
        f"| Model confidence | {cls_probs[burden_idx]*100:.1f}% |\n"
        f"| Mean seg uncertainty | {seg_unc_mean:.4f} |\n\n"
        f"**Burden probabilities:** {cls_bar}\n\n"
        f"---\n"
        f"*Research prototype. Not for medical diagnosis.*"
    )
    return tmp.name, report


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
DESCRIPTION = """
<div style="text-align:center; padding:1rem 0">
  <h1 style="font-size:2rem">🧠 Brain Tumor Segmentation</h1>
  <p style="color:#94a3b8">Swin-EVDNet v2 · 3D evidential deep learning · BraTS 2023</p>
  <div style="background:#7f1d1d;border-left:4px solid #ef4444;padding:0.7rem 1rem;
              border-radius:6px;margin:1rem auto;max-width:600px;text-align:left">
    ⚠️ <strong>Research prototype. Not for medical diagnosis.</strong>
    Do not upload real patient data to a public server.
  </div>
</div>
"""

with gr.Blocks(
    title="Brain Tumor Segmentation | Swin-EVDNet",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#0f172a",
        body_background_fill_dark="#0f172a",
        block_background_fill="#1e293b",
        block_background_fill_dark="#1e293b",
        block_border_color="#334155",
        block_border_color_dark="#334155",
        input_background_fill="#0f172a",
        input_background_fill_dark="#0f172a",
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_400",
    ),
    css="""
      body { font-family: 'Inter', sans-serif; }
      .gr-button-primary { font-weight: 600; border-radius: 8px; }
      footer { display: none !important; }
    """,
) as demo:
    gr.HTML(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload MRI volumes (.nii or .nii.gz)")
            t1_up    = gr.File(label="T1",    file_types=[".nii", ".gz"])
            t1ce_up  = gr.File(label="T1ce",  file_types=[".nii", ".gz"])
            t2_up    = gr.File(label="T2",    file_types=[".nii", ".gz"])
            flair_up = gr.File(label="FLAIR", file_types=[".nii", ".gz"])
            threshold_sl = gr.Slider(
                0.1, 0.9, value=SEG_THRESHOLD, step=0.05,
                label="Segmentation threshold",
            )
            run_btn = gr.Button("🔬 Run Segmentation", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Output")
            out_img  = gr.Image(label="Visualisation", type="filepath",
                                show_download_button=True)
            out_text = gr.Markdown(label="Report")

    run_btn.click(
        fn=run_segmentation,
        inputs=[t1_up, t1ce_up, t2_up, flair_up, threshold_sl],
        outputs=[out_img, out_text],
    )

    gr.Markdown("""
---
**Model:** Swin-EVDNet v2 3D · 4.17M parameters · Trained 25 epochs on BraTS 2023 Adult Glioma  
**Weights:** [aryan09103/swin-evdnet](https://huggingface.co/aryan09103/swin-evdnet)  
**Burden categories** are whole-tumour volume tertiles from the training split — not HGG/LGG grade.
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
