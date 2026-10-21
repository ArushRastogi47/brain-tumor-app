---
title: Brain Tumor Segmentation
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
python_version: "3.11"
---

# Brain Tumor Segmentation — Swin-EVDNet v2 3D

Streamlit app for evidential 3D brain tumour MRI segmentation using a
native 3D Swin-Transformer encoder (Swin-EVDNet v2).

Upload the four co-registered MRI modalities **(T1, T1ce, T2, FLAIR)** in
NIfTI format (`.nii` or `.nii.gz`) to receive:

- 🔴 Predicted tumour segmentation mask
- 🌡️ Per-voxel evidential uncertainty map
- 📊 Patient-level tumour-burden classification (low / medium / high)

**Research prototype. Not for medical diagnosis.**

---

## Model details

| Property | Value |
|---|---|
| Architecture | 3D SwinTransformer encoder + ASPP3D + evidential decoder |
| Input | 4 MRI channels — T1, T1ce, T2, FLAIR (96³ crop, 50 % overlap) |
| Output | Segmentation probability + uncertainty + 3-class burden |
| Training data | BraTS 2023 Adult Glioma (70 / 15 / 15 split) |
| Epochs | 25 |
| Framework | PyTorch + MONAI |

Weights: [Hugging Face model repo](https://huggingface.co/aryan09103/swin-evdnet)

---

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## API (FastAPI alternative)

See `results (1)/app.py` for a FastAPI version that saves prediction NIfTI
files and exposes `/predict` and `/health` endpoints.

---

*Tumour-burden categories (low / medium / high) are whole-tumour volume
tertiles from the training split. They are not equivalent to clinical
HGG/LGG grading and must not be used for any clinical decision.*
