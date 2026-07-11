import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from utils.hull_utils import *

st.set_page_config(page_title="HullGuard — Analysis", page_icon="📊", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Share+Tech+Mono&display=swap');
html,body,[class*="css"]{background:#050d1a!important;color:#c8dff0!important}
.stApp{background:#050d1a!important}
section[data-testid="stSidebar"]{background:#0a1628!important;border-right:1px solid #1a3a5c!important}
h1,h2,h3{font-family:'Orbitron',monospace!important;color:#00d4ff!important}
[data-testid="stMetric"]{background:#0d1f35!important;border:1px solid #1a3a5c!important;border-radius:8px!important;padding:12px 16px!important}
[data-testid="stMetricValue"]{color:#00d4ff!important;font-family:'Share Tech Mono',monospace!important}
.stTabs [data-baseweb="tab"]{font-family:'Orbitron',monospace!important;font-size:0.7rem!important;color:#5a7a96!important}
.stTabs [aria-selected="true"]{color:#00d4ff!important;border-bottom-color:#00d4ff!important}
.stProgress>div>div>div{background:linear-gradient(90deg,#00aacc,#00d4ff)!important}
</style>""", unsafe_allow_html=True)

nav_header("DAMAGE ANALYSIS", "PIXEL-LEVEL DAMAGE QUANTIFICATION & HEATMAP GENERATION", "📊")

# ── Retrieve or re-run ─────────────────────────────────────────────────────────
if "detections" not in st.session_state or "processed" not in st.session_state:
    st.warning("⚠ Please upload an image on the **Detection** page first.")
    uploaded = st.file_uploader("Or upload directly here:", type=["jpg","jpeg","png"])
    if uploaded:
        img_array = load_uploaded_image(uploaded)
        processed = preprocess_image(img_array)
        detections = run_mock_detection(processed["resized"])
    else:
        st.stop()
else:
    processed  = st.session_state["processed"]
    detections = st.session_state["detections"]
    img_array  = st.session_state["img_array"]

# ── Run analysis ───────────────────────────────────────────────────────────────
analysis = calculate_damage_percentage(processed["resized"], detections)
heatmap  = generate_heatmap(processed["resized"], analysis["damage_mask"])
annotated = draw_detections(processed["resized"], detections)

# ── Top metrics ────────────────────────────────────────────────────────────────
dmg  = analysis["damage_pct"]
sev  = analysis["severity_score"]
conf = analysis["mean_confidence"]

col1,col2,col3,col4,col5 = st.columns(5)
col1.metric("Damage Area %",     f"{dmg:.2f}%",    delta=None)
col2.metric("Severity Score",    f"{sev:.1f}/100")
col3.metric("Detection Accuracy",f"{conf:.1f}%")
col4.metric("Damage Regions",    str(analysis["num_detections"]))
col5.metric("Image Size",        f"{processed['resized'].shape[1]}×{processed['resized'].shape[0]}")

# ── Severity gauge ─────────────────────────────────────────────────────────────
gauge_colour = "#00ff9d" if sev < 20 else "#ffdd00" if sev < 50 else "#ff6b35" if sev < 75 else "#ff2d55"
st.markdown(f"""
<div style="background:#0d1f35;border:1px solid #1a3a5c;border-radius:10px;padding:16px 20px;margin:16px 0">
  <div style="font-family:Orbitron,monospace;font-size:0.72rem;color:#5a7a96;letter-spacing:0.1em;margin-bottom:8px">
    OVERALL SEVERITY INDEX
  </div>
  <div style="background:#0a1628;border-radius:4px;height:18px;overflow:hidden;border:1px solid #1a3a5c">
    <div style="width:{sev}%;height:100%;background:linear-gradient(90deg,{gauge_colour}88,{gauge_colour});
                border-radius:4px;transition:width 0.5s"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:6px;
              font-family:Share Tech Mono,monospace;font-size:0.72rem;color:#5a7a96">
    <span>0 — INTACT</span>
    <span style="color:{gauge_colour};font-weight:600">{sev:.1f} / 100</span>
    <span>100 — CATASTROPHIC</span>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🌡️  DAMAGE HEATMAP",
    "📐  PIXEL ANALYSIS",
    "📊  CLASS BREAKDOWN",
    "🔬  METHODOLOGY"
])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Original**")
        st.image(processed["resized"], use_container_width=True)
    with c2:
        st.markdown("**Annotated Detections**")
        st.image(annotated, use_container_width=True)
    with c3:
        st.markdown("**Damage Heatmap (Jet colormap)**")
        st.image(heatmap, use_container_width=True)

    st.markdown("**Damage mask (binary)**")
    st.image(analysis["damage_mask"], use_container_width=False, width=320, clamp=True)
    theory_box("Heatmap generation: damage bounding boxes → binary mask → Gaussian blur (σ=25) → "
               "cv2.applyColorMap(COLORMAP_JET) → alpha blend with original (α=0.45). "
               "Red/orange = high damage concentration. Blue/green = lower damage.")

with tab2:
    # Pixel intensity histogram of damaged vs undamaged regions
    resized_gray = cv2.cvtColor(processed["resized"], cv2.COLOR_RGB2GRAY)
    damage_mask  = analysis["damage_mask"]

    damaged_pixels   = resized_gray[damage_mask > 0]
    undamaged_pixels = resized_gray[damage_mask == 0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#050d1a")
    for ax in axes: ax.set_facecolor("#0d1f35")

    # Histogram comparison
    axes[0].hist(undamaged_pixels.flatten(), bins=64, color="#00aacc", alpha=0.7, label="Undamaged", density=True)
    axes[0].hist(damaged_pixels.flatten(),   bins=64, color="#ff6b35", alpha=0.7, label="Damaged",   density=True)
    axes[0].set_title("Pixel Intensity Distribution", color="#c8dff0", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].tick_params(colors="#5a7a96"); axes[0].set_xlabel("Intensity", color="#5a7a96")
    for spine in axes[0].spines.values(): spine.set_color("#1a3a5c")

    # Box area breakdown
    if analysis["box_areas"]:
        axes[1].bar(range(len(analysis["box_areas"])),
                    analysis["box_areas"], color="#00d4ff", alpha=0.85)
        axes[1].set_title("Per-detection Area (%)", color="#c8dff0", fontsize=10)
        axes[1].set_xlabel("Detection #", color="#5a7a96")
        axes[1].set_ylabel("% of image", color="#5a7a96")
        axes[1].tick_params(colors="#5a7a96")
        for spine in axes[1].spines.values(): spine.set_color("#1a3a5c")

    # Damage mask
    axes[2].imshow(damage_mask, cmap="hot", vmin=0, vmax=255)
    axes[2].set_title("Damage Mask (Binary)", color="#c8dff0", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Damaged pixels",   f"{(damage_mask>0).sum():,}")
    col2.metric("Total pixels",     f"{damage_mask.size:,}")
    col3.metric("Damaged pixel %",  f"{dmg:.2f}%")
    col4.metric("Mean intensity (damaged)", f"{damaged_pixels.mean():.1f}" if len(damaged_pixels) else "N/A")

with tab3:
    if analysis["class_stats"]:
        st.markdown("**Damage breakdown by class:**")
        class_stats = analysis["class_stats"]
        total_area  = sum(v["area_pct"] for v in class_stats.values())

        for cls, stats in class_stats.items():
            pct    = stats["area_pct"]
            ratio  = pct / max(total_area, 0.01)
            colour = {"Surface Crack":"#00C8FF","Deep Fracture":"#5050FF",
                      "Corrosion Patch":"#00B478","Deformation":"#FFA500",
                      "Breach / Hole":"#FF2828"}.get(cls, "#00d4ff")

            st.markdown(f"""
            <div style="background:#0d1f35;border:1px solid #1a3a5c;border-radius:8px;
                        padding:14px 16px;margin-bottom:10px">
              <div style="display:flex;justify-content:space-between;margin-bottom:8px">
                <span style="font-size:0.9rem;color:#c8dff0;font-weight:500">{cls}</span>
                <span style="font-family:Share Tech Mono,monospace;color:{colour}">{pct:.2f}% of image</span>
              </div>
              <div style="display:flex;gap:16px;font-size:0.78rem;color:#5a7a96;margin-bottom:8px">
                <span>Count: <b style="color:#c8dff0">{stats["count"]}</b></span>
                <span>Max confidence: <b style="color:{colour}">{stats["max_conf"]*100:.1f}%</b></span>
                <span>Edge density: <b style="color:#c8dff0">{stats["edge_density"]:.1f}%</b></span>
              </div>
              <div style="background:#0a1628;border-radius:3px;height:8px;border:1px solid #1a3a5c">
                <div style="width:{min(ratio*100,100):.1f}%;height:100%;background:{colour};border-radius:3px"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Pie chart
        fig2, ax2 = plt.subplots(figsize=(5, 5), facecolor="#050d1a")
        ax2.set_facecolor("#0d1f35")
        labels = list(class_stats.keys())
        sizes  = [v["area_pct"] for v in class_stats.values()]
        pie_colours = ["#00C8FF","#5050FF","#00B478","#FFA500","#FF2828"][:len(labels)]
        ax2.pie(sizes, labels=labels, colors=pie_colours, autopct="%1.1f%%",
                textprops={"color":"#c8dff0","fontsize":8})
        ax2.set_title("Damage class distribution", color="#c8dff0", fontsize=10)
        plt.tight_layout()
        buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", dpi=110, bbox_inches="tight"); buf2.seek(0)
        col_pie, _ = st.columns([1,1])
        with col_pie: st.image(buf2)
        plt.close(fig2)

with tab4:
    theory_box("Damage percentage is calculated by: (1) creating a binary damage mask from all detection "
               "bounding boxes, (2) computing the union of masked pixels, (3) dividing by total image pixels. "
               "Severity score = mean(confidence × severity_weight × 100) over all detections, capped at 100. "
               "Edge density within each ROI (Canny edges / ROI pixels) provides a secondary crack-density metric.")

    st.markdown("""
    <div style="background:#0d1f35;border:1px solid #1a3a5c;border-radius:8px;padding:16px;
                font-family:Share Tech Mono,monospace;font-size:0.78rem;color:#5a7a96;line-height:2">
    DAMAGE_PCT = (union_mask_pixels / total_pixels) × 100<br>
    SEVERITY   = mean(confidence_i × severity_weight_i) × 100<br>
    EDGE_DENSITY = Canny_edge_pixels / ROI_pixels × 100<br>
    SEVERITY_WEIGHTS: Surface Crack=0.20, Corrosion=0.35, Deformation=0.45, Deep Fracture=0.55, Breach=0.90
    </div>""", unsafe_allow_html=True)

# ── Store in session ───────────────────────────────────────────────────────────
st.session_state["analysis"] = analysis
