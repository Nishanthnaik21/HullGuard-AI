import streamlit as st

st.set_page_config(
    page_title="HullGuard AI — Marine Safety Monitor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

:root {
    --navy:    #050d1a;
    --deep:    #0a1628;
    --panel:   #0d1f35;
    --border:  #1a3a5c;
    --accent:  #00d4ff;
    --warn:    #ff6b35;
    --danger:  #ff2d55;
    --safe:    #00ff9d;
    --text:    #c8dff0;
    --dim:     #5a7a96;
}

html, body, [class*="css"] {
    background-color: var(--navy) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

.stApp { background: var(--navy) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--deep) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Headers */
h1,h2,h3 { font-family: 'Orbitron', monospace !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--panel) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #00aacc, #0077aa) !important;
    color: white !important;
    border: none !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 4px !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #00ccee, #0099cc) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.3) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] { color: var(--dim) !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 1.4rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.72rem !important;
    color: var(--dim) !important;
    letter-spacing: 0.08em !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Progress bars */
.stProgress > div > div { background: var(--border) !important; }
.stProgress > div > div > div { background: linear-gradient(90deg, #00aacc, #00d4ff) !important; }

/* Selectbox / slider */
.stSelectbox > div, .stSlider { color: var(--text) !important; }

/* Info / warning boxes */
.stAlert { border-radius: 6px !important; border: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px 0; border-bottom:1px solid #1a3a5c; margin-bottom:24px">
  <div style="font-family:'Orbitron',monospace; font-size:2rem; font-weight:800;
              background:linear-gradient(90deg,#00d4ff,#00ff9d); -webkit-background-clip:text;
              -webkit-text-fill-color:transparent; letter-spacing:0.05em">
    HULLGUARD AI
  </div>
  <div style="font-family:'Share Tech Mono',monospace; font-size:0.85rem; color:#5a7a96;
              letter-spacing:0.15em; margin-top:4px">
    MARINE HULL DAMAGE DETECTION & SURVIVAL PREDICTION SYSTEM
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.9rem; color:#00d4ff;
                letter-spacing:0.1em; margin-bottom:16px">⚙ SYSTEM CONFIG</div>
    """, unsafe_allow_html=True)

    model_type = st.selectbox("Detection Model", ["YOLOv8n (Fast)", "YOLOv8s (Balanced)", "YOLOv8m (Accurate)"])
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.95, 0.35, 0.05)
    iou_threshold  = st.slider("IoU Threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
    preprocess_mode = st.selectbox("Preprocessing Mode", ["Standard", "Enhanced (CLAHE)", "Edge-boosted"])

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.9rem; color:#00d4ff;
                letter-spacing:0.1em; margin-bottom:12px">📋 PIPELINE STATUS</div>
    """, unsafe_allow_html=True)

    stages = [
        ("Input Module",       True),
        ("Preprocessing",      True),
        ("AI Detection",       True),
        ("Damage Analysis",    True),
        ("Survival Predictor", True),
        ("Dashboard Output",   True),
    ]
    for name, ok in stages:
        colour = "#00ff9d" if ok else "#ff6b35"
        symbol = "●" if ok else "○"
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:5px 0;font-family:'Share Tech Mono',monospace;font-size:0.8rem">
          <span style="color:#c8dff0">{name}</span>
          <span style="color:{colour}">{symbol} READY</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#5a7a96; line-height:1.6">
    <b style="color:#00d4ff">Course:</b> 22AIM61 — DIP<br>
    <b style="color:#00d4ff">Domain:</b> Marine Safety AI<br>
    <b style="color:#00d4ff">Model:</b> YOLOv8 + OpenCV<br>
    <b style="color:#00d4ff">Framework:</b> Streamlit
    </div>""", unsafe_allow_html=True)

# ── Navigation tiles ───────────────────────────────────────────────────────────
cols = st.columns(4)
nav_items = [
    ("🔍", "Detect Damage",   "pages/1_Detection.py",    "Upload hull image and run AI detection"),
    ("📊", "Damage Analysis", "pages/2_Analysis.py",     "Pixel-level damage percentage & heatmap"),
    ("⏱️", "Survival Predictor","pages/3_Survival.py",   "Estimate survival time from severity rules"),
    ("📡", "Live Dashboard",  "pages/4_Dashboard.py",    "Full monitoring dashboard & export"),
]
for col, (icon, title, page, desc) in zip(cols, nav_items):
    with col:
        st.markdown(f"""
        <div style="background:#0d1f35; border:1px solid #1a3a5c; border-radius:10px;
                    padding:18px 14px; text-align:center; min-height:120px;
                    transition:border-color 0.2s; cursor:pointer">
          <div style="font-size:1.8rem; margin-bottom:8px">{icon}</div>
          <div style="font-family:'Orbitron',monospace; font-size:0.75rem; font-weight:600;
                      color:#00d4ff; letter-spacing:0.08em; margin-bottom:6px">{title.upper()}</div>
          <div style="font-size:0.75rem; color:#5a7a96; line-height:1.4">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Architecture diagram ───────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'Orbitron',monospace; font-size:0.85rem; color:#00d4ff;
            letter-spacing:0.12em; margin-bottom:14px">◈ SYSTEM ARCHITECTURE</div>
""", unsafe_allow_html=True)

arch_stages = [
    ("#1a3a5c", "📥", "INPUT MODULE",        "Web upload\nJPG/PNG/TIFF"),
    ("#0d2a1a", "⚙️", "PREPROCESS",          "Resize → Filter\nEdge Detect"),
    ("#1a2a0d", "🤖", "YOLO DETECTION",      "Crack & damage\nbox detection"),
    ("#1a1a0d", "📐", "DAMAGE ANALYSIS",     "Pixel ratio\n% calculation"),
    ("#1a0d0d", "⏱️", "SURVIVAL PREDICT",    "Severity rules\ntime estimate"),
    ("#0d1a1a", "📊", "DASHBOARD OUTPUT",    "Viz + alerts\nCSV export"),
]

arch_cols = st.columns(len(arch_stages))
for col, (bg, icon, title, desc) in zip(arch_cols, arch_stages):
    with col:
        st.markdown(f"""
        <div style="background:{bg}; border:1px solid #1a3a5c; border-radius:8px;
                    padding:14px 10px; text-align:center; position:relative">
          <div style="font-size:1.4rem">{icon}</div>
          <div style="font-family:'Orbitron',monospace; font-size:0.65rem; font-weight:600;
                      color:#00d4ff; margin:6px 0 4px 0; letter-spacing:0.06em">{title}</div>
          <div style="font-family:'Share Tech Mono',monospace; font-size:0.68rem;
                      color:#5a7a96; white-space:pre-line; line-height:1.4">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#1a3a5c; font-size:0.75rem; margin-top:6px;
            font-family:'Share Tech Mono',monospace; letter-spacing:0.2em">
  ── → ── → ── → ── → ── → ──
</div>""", unsafe_allow_html=True)

# ── Tech stack ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'Orbitron',monospace; font-size:0.85rem; color:#00d4ff;
            letter-spacing:0.12em; margin-bottom:14px">◈ TECHNOLOGY STACK</div>
""", unsafe_allow_html=True)

tech = [
    ("Python 3.10+",     "Core language",              "#00aacc"),
    ("OpenCV 4.8",       "Preprocessing & analysis",   "#00ccaa"),
    ("YOLOv8 (Ultralytics)", "AI damage detection",    "#cc6600"),
    ("PyTorch 2.x",      "DL framework",               "#ee4422"),
    ("Streamlit 1.28",   "Web dashboard",              "#ff4b6e"),
    ("NumPy / Pandas",   "Data computation",           "#7744cc"),
    ("Matplotlib",       "Visualisation",              "#4488ff"),
    ("Pillow",           "Image I/O",                  "#44aa44"),
]
tech_cols = st.columns(4)
for i, (name, role, colour) in enumerate(tech):
    with tech_cols[i % 4]:
        st.markdown(f"""
        <div style="background:#0d1f35; border-left:3px solid {colour};
                    border-radius:0 6px 6px 0; padding:10px 12px; margin-bottom:8px">
          <div style="font-size:0.82rem; font-weight:500; color:#c8dff0">{name}</div>
          <div style="font-size:0.72rem; color:#5a7a96">{role}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""<div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem;
color:#5a7a96; text-align:center; letter-spacing:0.1em">
HULLGUARD AI v1.0 · MARINE SAFETY MONITORING · COURSE 22AIM61
</div>""", unsafe_allow_html=True)
