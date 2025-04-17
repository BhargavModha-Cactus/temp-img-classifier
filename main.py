from __future__ import annotations
import json, warnings
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import contextlib

st.set_page_config(page_title="DocFigure Classifier", layout="centered")

CKPT_PATH = Path("best_ckpt.pth")
IDX2LBL_PATH = Path("idx2label.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FALLBACK_CLASSES = [
    "Line graph", "Natural image", "Table", "3D object", "Bar plot", "Scatter plot",
    "Medical image", "Sketch", "Geographic map", "Flow chart", "Heat map", "Mask",
    "Block diagram", "Venn diagram", "Confusion matrix", "Histogram", "Box plot",
    "Vector plot", "Pie chart", "Surface plot", "Algorithm", "Contour plot",
    "Tree diagram", "Bubble chart", "Polar plot", "Area chart", "Pareto chart",
    "Radar chart",
]
COMPLEXITY_MAP = {
    # Simple
    **{k: "Simple" for k in ["Natural image", "Sketch", "Medical image"]},
    # Moderate
    **{k: "Moderately complex" for k in [
        "Bar plot", "Line graph", "Scatter plot", "Pie chart", "Histogram",
        "Box plot", "Heat map", "Table", "Geographic map"]},
    # Everything else
    **{k: "Extremely complex" for k in FALLBACK_CLASSES if k not in [
        "Natural image", "Sketch", "Medical image",
        "Bar plot", "Line graph", "Scatter plot", "Pie chart", "Histogram",
        "Box plot", "Heat map", "Table", "Geographic map"]}
}


def build_model(num_classes: int) -> nn.Module:
    mdl = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    in_f = mdl.classifier[0].in_features
    mdl.classifier = nn.Sequential(
        nn.Linear(in_f, 1024), nn.Hardswish(), nn.Dropout(0.2),
        nn.Linear(1024, num_classes)
    )
    return mdl


def load_idx2label() -> dict[int, str]:
    if IDX2LBL_PATH.exists():
        with open(IDX2LBL_PATH, encoding="utf-8") as f:
            js = json.load(f)
        return {int(k): v for k, v in js.items()}
    warnings.warn("idx2label.json missing – falling back to hard‑coded 28‑class list.")
    return {i: l for i, l in enumerate(FALLBACK_CLASSES)}


def _st_cache():
    return st.cache_resource if hasattr(st, "cache_resource") else st.experimental_singleton


@_st_cache()
def load_model(ckpt: Path, num_classes: int):
    if not ckpt.exists():
        st.error(f"Checkpoint '{ckpt}' not found.");
        st.stop()
    mdl = build_model(num_classes)
    state = torch.load(ckpt, map_location="cpu")
    # quick sanity‑check: class count matches idx2label
    expect = mdl.classifier[-1].weight.shape[0]
    got = state["classifier.3.weight"].shape[0]
    if expect != got:
        st.error(f"Checkpoint expects {got} classes but idx2label.json has {expect}.")
        st.stop()
    mdl.load_state_dict(state, strict=True)
    mdl.eval().to(DEVICE)
    return mdl


@_st_cache()
def transform():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def predict(mdl: nn.Module, img: Image.Image, idx2lbl: dict[int, str]):
    x = transform()(img).unsqueeze(0).to(DEVICE)

    #  autocast only on CUDA; otherwise do nothing  ──────────────────────────
    amp_ctx = torch.amp.autocast(device_type="cuda") if DEVICE == "cuda" \
        else contextlib.nullcontext()

    with amp_ctx, torch.inference_mode():
        logits = mdl(x)  # may be fp16 (CUDA) or fp32/bf16 (CPU)
    probs = F.softmax(logits.float(), 1)[0].cpu().numpy()  # ← cast to fp32
    return probs, int(probs.argmax())


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("📊 DocFigure Classifier")

idx2lbl = load_idx2label()
lbl2cplx = {l: COMPLEXITY_MAP.get(l, "Simple") for l in idx2lbl.values()}
model = load_model(CKPT_PATH, num_classes=len(idx2lbl))

img_file = st.file_uploader("Upload a figure image",
                            type=["png", "jpg", "jpeg", "bmp", "gif"])

if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running inference…"):
        probs, pred_idx = predict(model, img, idx2lbl)
        subclass = idx2lbl[pred_idx]
        complexity = lbl2cplx[subclass]

    st.success(f"### {subclass}\n**Complexity:** {complexity}")

    k = 5
    topk_idx = probs.argsort()[-k:][::-1]
    df = pd.DataFrame({
        "label": [idx2lbl[i] for i in topk_idx],
        "probability": probs[topk_idx]
    })
    st.altair_chart(
        alt.Chart(df).mark_bar().encode(
            x=alt.X("probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("label:N", sort="-x", title="")
        ).properties(width=500),
        use_container_width=True
    )

    with st.expander("Raw confidences JSON"):
        st.json({idx2lbl[i]: float(p) for i, p in enumerate(probs)})

else:
    st.info("Upload a PNG / JPG image to begin.")
