import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from utils.hull_utils import *

st.set_page_config(page_title="HullGuard — Detection", page_icon="🔍", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Share+Tech+Mono&display=swap');
html,body,[class*="css"]{background:#050d1a!important;color:#c8dff0!important}
.stApp{background:#050d1a!important}
section[data-testid="stSidebar"]{background:#0a1628!important;border-right:1px solid #1a3a5c!important}
h1,h2,h3{font-family:'Orbitron',monospace!important;color:#00d4ff!important}
[data-testid="stMetric"]{background:#0d1f35!important;border:1px solid #1a3a5c!important;border-radius:8px!important;padding:12px 16px!important}
[data-testid="stMetricValue"]{color:#00d4ff!important;font-family:'Share Tech Mono',monospace!important}
.stButton button{background:linear-gradient(135deg,#00aacc,#0077aa)!important;color:white!important;border:none!important;font-family:'Orbitron',monospace!important;font-size:0.75rem!important;letter-spacing:0.1em!important;border-radius:4px!important}
.stTabs [data-baseweb="tab"]{font-family:'Orbitron',monospace!important;font-size:0.7rem!important;color:#5a7a96!important;letter-spacing:0.08em!important}
.stTabs [aria-selected="true"]{color:#00d4ff!important;border-bottom-color:#00d4ff!important}
</style>""", unsafe_allow_html=True)

nav_header("DAMAGE DETECTION", "AI-POWERED HULL CRACK & DEFECT IDENTIFICATION", "🔍")

# ── Sidebar config ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Orbitron,monospace;font-size:0.85rem;color:#00d4ff;letter-spacing:0.1em;margin-bottom:12px">⚙ DETECTION SETTINGS</div>', unsafe_allow_html=True)
    conf_threshold  = st.slider("Confidence Threshold", 0.10, 0.95, 0.35, 0.05)
    iou_threshold   = st.slider("IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
    preprocess_mode = st.selectbox("Preprocessing Mode", ["Standard","Enhanced (CLAHE)","Edge-boosted"])
    show_steps      = st.checkbox("Show preprocessing steps", value=True)

    st.markdown("---")
    theory_box("YOLOv8 divides the image into a grid. Each cell predicts bounding boxes "
               "and class probabilities simultaneously. NMS removes overlapping boxes "
               "keeping only the highest-confidence detection per object.")

# ── Main upload ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload hull image (JPG / PNG / TIFF)", type=["jpg","jpeg","png","tiff","tif"])

if uploaded is None:
    st.markdown("""
    <div style="background:#0d1f35;border:2px dashed #1a3a5c;border-radius:12px;
                padding:40px;text-align:center;margin-top:20px">
      <div style="font-size:3rem;margin-bottom:12px">🚢</div>
      <div style="font-family:Orbitron,monospace;font-size:0.9rem;color:#5a7a96;letter-spacing:0.1em">
        UPLOAD A HULL IMAGE TO BEGIN ANALYSIS
      </div>
      <div style="font-size:0.78rem;color:#2a4a6c;margin-top:8px">
        Supports: JPG, PNG, TIFF · Max size: 200MB
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

img_array = load_uploaded_image(uploaded)
if img_array is None:
    st.error("Could not read image."); st.stop()

# ── Run pipeline ───────────────────────────────────────────────────────────────
with st.spinner("🔄 Running preprocessing pipeline..."):
    processed = preprocess_image(img_array, preprocess_mode)

with st.spinner("🤖 Running YOLO detection..."):
    detections = run_mock_detection(processed["resized"], conf_threshold)

annotated = draw_detections(processed["resized"], detections)

# ── Top metrics ────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
mean_conf = sum(d["confidence"] for d in detections)/max(len(detections),1)
c1.metric("Detections Found",   str(len(detections)))
c2.metric("Mean Confidence",    f"{mean_conf*100:.1f}%")
c3.metric("Model",              "YOLOv8 (Mock)")
c4.metric("Input Resolution",   f"{processed['resized'].shape[1]}×{processed['resized'].shape[0]}")
c5.metric("Preprocessing",      preprocess_mode.split()[0])

st.markdown("<br>", unsafe_allow_html=True)

# ── Detection results ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🖼️  ANNOTATED OUTPUT", "⚙️  PREPROCESSING STEPS", "📋  DETECTION TABLE"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Input**")
        st.image(img_array, use_container_width=True)
    with col2:
        st.markdown(f"**YOLO Detection Result — {len(detections)} damage region(s) found**")
        st.image(annotated, use_container_width=True)

    # Legend
    st.markdown("**Detection classes:**")
    leg_cols = st.columns(5)
    colours = {"Surface Crack":"#00C8FF","Deep Fracture":"#5050FF","Corrosion Patch":"#00B478","Deformation":"#FFA500","Breach / Hole":"#FF2828"}
    for col, (name, clr) in zip(leg_cols, colours.items()):
        col.markdown(f'<span style="background:{clr}33;border:1px solid {clr};border-radius:4px;'
                     f'padding:3px 8px;font-size:0.75rem;color:{clr}">{name}</span>', unsafe_allow_html=True)

with tab2:
    if show_steps:
        steps = [
            (processed["original"],  "1. Original"),
            (processed["resized"],   "2. Resized (640×640)"),
            (processed["gray"],      "3. Grayscale"),
            (processed["denoised"],  "4. Denoised"),
            (processed["enhanced"],  f"5. {preprocess_mode}"),
            (processed["edges"],     "6. Canny Edges"),
            (processed["morph"],     "7. Morphological Close"),
        ]
        for i in range(0, len(steps), 4):
            batch = steps[i:i+4]
            cols  = st.columns(len(batch))
            for col, (img, label) in zip(cols, batch):
                with col:
                    st.markdown(f"**{label}**")
                    st.image(img, use_container_width=True, clamp=True)
        theory_box("Pipeline: Resize (640×640) → Grayscale → FastNLM Denoise → "
                   f"CLAHE/EqualiseHist ({preprocess_mode}) → Canny Edge Detection → "
                   "Morphological Closing (3×3 kernel, 2 iterations) → YOLO inference on enhanced RGB.")

with tab3:
    if detections:
        st.markdown("**Individual detection results:**")
        for i, det in enumerate(detections):
            x1,y1,x2,y2 = det["box"]
            w_box = x2-x1; h_box = y2-y1
            cols = st.columns([1,2,2,2,2,2])
            cols[0].markdown(f'<div style="font-family:monospace;color:#5a7a96;padding-top:8px">#{i+1}</div>', unsafe_allow_html=True)
            cols[1].markdown(f'<div style="color:#00d4ff;font-size:0.85rem;padding-top:8px">{det["class_name"]}</div>', unsafe_allow_html=True)
            cols[2].metric("Confidence", f"{det['confidence']*100:.1f}%")
            cols[3].metric("Box (px)", f"{w_box}×{h_box}")
            cols[4].metric("Severity weight", f"{det['severity_weight']:.2f}")
            cols[5].metric("Area (px²)", f"{w_box*h_box:,}")
    else:
        st.success("✅ No damage regions detected above confidence threshold.")

# ── Store in session ───────────────────────────────────────────────────────────
st.session_state["detections"]     = detections
st.session_state["processed"]      = processed
st.session_state["img_array"]      = img_array
st.session_state["conf_threshold"] = conf_threshold

st.markdown("---")
st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.72rem;color:#2a4a6c;text-align:center">→ Proceed to DAMAGE ANALYSIS page for pixel-level breakdown</div>', unsafe_allow_html=True)
