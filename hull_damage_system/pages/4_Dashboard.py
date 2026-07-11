import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import io, json
from datetime import datetime
from utils.hull_utils import *

st.set_page_config(page_title="HullGuard — Dashboard", page_icon="📡", layout="wide")
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
.stButton button{background:linear-gradient(135deg,#00aacc,#0077aa)!important;color:white!important;border:none!important;font-family:'Orbitron',monospace!important;font-size:0.75rem!important;border-radius:4px!important}
.stDataFrame{background:#0d1f35!important;}
</style>""", unsafe_allow_html=True)

nav_header("MONITORING DASHBOARD", "FULL SYSTEM REPORT · EXPORT · ALERT LOG", "📡")

# ── Get all session data or demo data ──────────────────────────────────────────
has_data = ("analysis" in st.session_state and "detections" in st.session_state)

if has_data:
    analysis   = st.session_state["analysis"]
    detections = st.session_state["detections"]
    processed  = st.session_state["processed"]
    img_array  = st.session_state["img_array"]
else:
    st.info("🔄 Running demo analysis for dashboard preview...")
    img_array = np.zeros((256,256,3),dtype=np.uint8)
    cv2.rectangle(img_array,(30,30),(200,200),(80,80,80),-1)
    cv2.putText(img_array,"DEMO",(70,140),cv2.FONT_HERSHEY_SIMPLEX,2,(150,150,150),3)
    processed  = preprocess_image(img_array)
    detections = run_mock_detection(processed["resized"])
    analysis   = calculate_damage_percentage(processed["resized"], detections)

vessel_type = st.sidebar.selectbox("Vessel Type", ["Cargo","Tanker","Passenger","Naval","Fishing","Recreational"])
vessel_age  = st.sidebar.slider("Vessel Age (years)", 0, 50, 10)
vessel_name = st.sidebar.text_input("Vessel Name / IMO", "MV DEMO-001")
report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

prediction = predict_survival(analysis["damage_pct"], analysis["severity_score"], vessel_age, vessel_type)
colour     = prediction["colour"]
risk       = prediction["risk_level"]

# ── Status header strip ────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:8px;
            background:#0d1f35;border:1px solid #1a3a5c;border-radius:10px;
            padding:14px 16px;margin-bottom:20px">
  <div style="text-align:center">
    <div style="font-family:Share Tech Mono,monospace;font-size:0.68rem;color:#5a7a96">VESSEL</div>
    <div style="font-size:0.85rem;color:#c8dff0;font-weight:500">{vessel_name}</div>
  </div>
  <div style="text-align:center">
    <div style="font-family:Share Tech Mono,monospace;font-size:0.68rem;color:#5a7a96">TYPE</div>
    <div style="font-size:0.85rem;color:#c8dff0">{vessel_type}</div>
  </div>
  <div style="text-align:center">
    <div style="font-family:Share Tech Mono,monospace;font-size:0.68rem;color:#5a7a96">DAMAGE %</div>
    <div style="font-family:Share Tech Mono,monospace;font-size:0.9rem;color:#ff6b35">{analysis['damage_pct']:.2f}%</div>
  </div>
  <div style="text-align:center">
    <div style="font-family:Share Tech Mono,monospace;font-size:0.68rem;color:#5a7a96">SEVERITY</div>
    <div style="font-family:Share Tech Mono,monospace;font-size:0.9rem;color:#ffdd00">{analysis['severity_score']:.1f}/100</div>
  </div>
  <div style="text-align:center">
    <div style="font-family:Share Tech Mono,monospace;font-size:0.68rem;color:#5a7a96">RISK</div>
    <div style="font-family:Orbitron,monospace;font-size:0.8rem;font-weight:700;color:{colour}">{risk}</div>
  </div>
  <div style="text-align:center">
    <div style="font-family:Share Tech Mono,monospace;font-size:0.68rem;color:#5a7a96">SURVIVAL</div>
    <div style="font-family:Share Tech Mono,monospace;font-size:0.9rem;color:{colour}">{prediction['survival_str']}</div>
  </div>
  <div style="text-align:center">
    <div style="font-family:Share Tech Mono,monospace;font-size:0.68rem;color:#5a7a96">TIMESTAMP</div>
    <div style="font-size:0.7rem;color:#5a7a96">{report_time[:16]}</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Alert banner ───────────────────────────────────────────────────────────────
alert_levels = ["MINIMAL","LOW","MODERATE","ELEVATED","HIGH","CRITICAL","EXTREME"]
if alert_levels.index(risk) >= 3:
    st.markdown(f"""
    <div style="background:{colour}15;border:1.5px solid {colour};border-radius:8px;
                padding:14px 18px;margin-bottom:16px;display:flex;align-items:center;gap:12px;
                animation:none">
      <div style="font-size:1.4rem">{"🆘" if alert_levels.index(risk) >= 5 else "⚠️"}</div>
      <div>
        <div style="font-family:Orbitron,monospace;font-size:0.82rem;color:{colour};
                    font-weight:700;letter-spacing:0.1em">ALERT: {risk} RISK LEVEL</div>
        <div style="font-family:Share Tech Mono,monospace;font-size:0.78rem;
                    color:#c8dff0;margin-top:3px">{prediction['action']}</div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── Main dashboard tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🖥️  FULL REPORT",
    "📊  VISUAL ANALYTICS",
    "📋  DETECTION LOG",
    "💾  EXPORT"
])

with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        annotated = draw_detections(processed["resized"], detections)
        heatmap   = generate_heatmap(processed["resized"], analysis["damage_mask"])
        st.markdown("**Detection Output**")
        st.image(annotated, use_container_width=True)
        st.markdown("**Damage Heatmap**")
        st.image(heatmap, use_container_width=True)

    with c2:
        st.markdown(f"""
        <div style="background:#0d1f35;border:1px solid #1a3a5c;border-radius:10px;padding:20px">
          <div style="font-family:Orbitron,monospace;font-size:0.8rem;color:#00d4ff;
                      letter-spacing:0.1em;margin-bottom:16px">◈ FULL ANALYSIS REPORT</div>

          <div style="font-family:Share Tech Mono,monospace;font-size:0.78rem;
                      line-height:2.2;color:#c8dff0">
            <div style="color:#5a7a96">VESSEL ──────────────────────────</div>
            <div>Name: <span style="color:#00d4ff">{vessel_name}</span></div>
            <div>Type: <span style="color:#00d4ff">{vessel_type}</span></div>
            <div>Age: <span style="color:#00d4ff">{vessel_age} years</span></div>

            <div style="color:#5a7a96;margin-top:8px">DETECTION ──────────────────────</div>
            <div>Regions found: <span style="color:#ffdd00">{analysis['num_detections']}</span></div>
            <div>Damage area: <span style="color:#ff6b35">{analysis['damage_pct']:.2f}%</span></div>
            <div>Mean confidence: <span style="color:#00d4ff">{analysis['mean_confidence']:.1f}%</span></div>
            <div>Severity score: <span style="color:#ff6b35">{analysis['severity_score']:.1f}/100</span></div>

            <div style="color:#5a7a96;margin-top:8px">PREDICTION ─────────────────────</div>
            <div>Risk level: <span style="color:{colour};font-weight:700">{risk}</span></div>
            <div>Base survival: <span style="color:{colour}">{prediction['base_hours']}h</span></div>
            <div>Adjusted survival: <span style="color:{colour};font-weight:700">{prediction['survival_str']}</span></div>
            <div>Action required: <span style="color:#c8dff0">{prediction['action']}</span></div>

            <div style="color:#5a7a96;margin-top:8px">REPORT ─────────────────────────</div>
            <div>Generated: <span style="color:#5a7a96">{report_time}</span></div>
            <div>Model: <span style="color:#00d4ff">YOLOv8 (Mock demo)</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Damage classes
        if analysis["class_stats"]:
            st.markdown("<br>**Damage class summary:**", unsafe_allow_html=True)
            for cls, stats in analysis["class_stats"].items():
                clr2 = {"Surface Crack":"#00C8FF","Deep Fracture":"#5050FF",
                         "Corrosion Patch":"#00B478","Deformation":"#FFA500",
                         "Breach / Hole":"#FF2828"}.get(cls, "#00d4ff")
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            background:#0a1628;border-left:2px solid {clr2};
                            border-radius:0 4px 4px 0;padding:6px 10px;margin-bottom:4px;font-size:0.78rem">
                  <span style="color:#c8dff0">{cls}</span>
                  <span style="color:{clr2};font-family:Share Tech Mono,monospace">
                    ×{stats['count']} | {stats['area_pct']:.2f}% | {stats['max_conf']*100:.0f}% conf
                  </span>
                </div>""", unsafe_allow_html=True)

with tab2:
    # Multi-panel analytics chart
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), facecolor="#050d1a")
    for ax in axes.flat: ax.set_facecolor("#0d1f35")

    # 1. Damage gauge (polar)
    ax0 = axes[0, 0]
    sev = analysis["severity_score"]
    gauge_data = [sev, 100-sev]
    ax0.pie(gauge_data, startangle=90, colors=[colour,"#1a3a5c"],
            wedgeprops=dict(width=0.35, edgecolor="#050d1a"))
    ax0.text(0, 0, f"{sev:.0f}", ha="center", va="center",
             fontsize=20, color=colour, fontfamily="monospace", fontweight="bold")
    ax0.text(0, -0.5, "SEVERITY", ha="center", color="#5a7a96", fontsize=8)
    ax0.set_title("Severity Index", color="#c8dff0", fontsize=9)

    # 2. Detection confidence bar
    if detections:
        names  = [f"#{i+1} {d['class_name'][:8]}" for i,d in enumerate(detections)]
        confs  = [d["confidence"]*100 for d in detections]
        bar_colours = [{"Surface Crack":"#00C8FF","Deep Fracture":"#5050FF",
                         "Corrosion Patch":"#00B478","Deformation":"#FFA500",
                         "Breach / Hole":"#FF2828"}.get(d["class_name"],"#00d4ff")
                        for d in detections]
        axes[0,1].barh(names, confs, color=bar_colours, alpha=0.85)
        axes[0,1].set_xlabel("Confidence %", color="#5a7a96")
        axes[0,1].set_title("Detection Confidence", color="#c8dff0", fontsize=9)
        axes[0,1].tick_params(colors="#5a7a96"); axes[0,1].set_xlim(0,100)
        for sp in axes[0,1].spines.values(): sp.set_color("#1a3a5c")

    # 3. Severity weight breakdown
    sw_names = list(CLASS_SEVERITY.keys())
    sw_vals  = list(CLASS_SEVERITY.values())
    sw_clrs  = ["#00C8FF","#5050FF","#00B478","#FFA500","#FF2828"]
    axes[0,2].bar(range(len(sw_names)), sw_vals, color=sw_clrs, alpha=0.85)
    axes[0,2].set_xticks(range(len(sw_names)))
    axes[0,2].set_xticklabels([n.split()[0] for n in sw_names], fontsize=7, rotation=15)
    axes[0,2].set_title("Severity Weights by Class", color="#c8dff0", fontsize=9)
    axes[0,2].tick_params(colors="#5a7a96")
    axes[0,2].set_ylim(0,1)
    for sp in axes[0,2].spines.values(): sp.set_color("#1a3a5c")

    # 4. Survival vs damage
    dmg_r = np.linspace(0, 100, 200)
    surv  = [predict_survival(d, d*0.6, vessel_age, vessel_type)["survival_hours"] for d in dmg_r]
    axes[1,0].plot(dmg_r, surv, color="#00d4ff", linewidth=2)
    axes[1,0].axvline(analysis["damage_pct"], color="#ffdd00", linestyle="--", alpha=0.8)
    axes[1,0].fill_between(dmg_r, 0, surv, alpha=0.15, color="#00d4ff")
    axes[1,0].set_title(f"Survival vs Damage % ({vessel_type})", color="#c8dff0", fontsize=9)
    axes[1,0].set_xlabel("Damage %", color="#5a7a96"); axes[1,0].set_ylabel("Hours", color="#5a7a96")
    axes[1,0].tick_params(colors="#5a7a96")
    for sp in axes[1,0].spines.values(): sp.set_color("#1a3a5c")

    # 5. Heatmap
    axes[1,1].imshow(cv2.GaussianBlur(analysis["damage_mask"],(51,51),0), cmap="hot")
    axes[1,1].set_title("Damage Concentration Heatmap", color="#c8dff0", fontsize=9); axes[1,1].axis("off")

    # 6. Risk level gauge
    risk_order = ["MINIMAL","LOW","MODERATE","ELEVATED","HIGH","CRITICAL","EXTREME"]
    risk_idx   = risk_order.index(risk)
    risk_clrs  = [RISK_COLOURS[r] for r in risk_order]
    axes[1,2].bar(range(7), [1]*7, color=[c+"44" for c in risk_clrs], edgecolor=risk_clrs)
    axes[1,2].bar(risk_idx, 1.2, color=colour, alpha=0.9, zorder=3)
    axes[1,2].set_xticks(range(7))
    axes[1,2].set_xticklabels([r[:3] for r in risk_order], fontsize=8)
    axes[1,2].set_title("Risk Level Indicator", color="#c8dff0", fontsize=9)
    axes[1,2].tick_params(colors="#5a7a96"); axes[1,2].set_ylim(0,1.5)
    for sp in axes[1,2].spines.values(): sp.set_color("#1a3a5c")

    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); buf.seek(0)
    st.image(buf, use_container_width=True); plt.close(fig)

with tab3:
    if detections:
        df_data = [{
            "ID":         i+1,
            "Class":      d["class_name"],
            "Confidence": f"{d['confidence']*100:.1f}%",
            "Box x1":     d["box"][0], "Box y1": d["box"][1],
            "Box x2":     d["box"][2], "Box y2": d["box"][3],
            "Width (px)": d["box"][2]-d["box"][0],
            "Height (px)":d["box"][3]-d["box"][1],
            "Area (px²)": (d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1]),
            "Sev. Weight":d["severity_weight"],
        } for i,d in enumerate(detections)]
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("### 💾 Export Report")
    report = {
        "vessel": {"name": vessel_name, "type": vessel_type, "age": vessel_age},
        "timestamp": report_time,
        "detection": {
            "num_detections":    analysis["num_detections"],
            "damage_pct":        analysis["damage_pct"],
            "severity_score":    analysis["severity_score"],
            "mean_confidence":   analysis["mean_confidence"],
            "class_breakdown":   {k: {kk: (float(vv) if isinstance(vv,float) else vv)
                                      for kk,vv in v.items() if kk != "edge_density"}
                                  for k,v in analysis["class_stats"].items()},
        },
        "prediction": {
            "risk_level":      prediction["risk_level"],
            "survival_hours":  prediction["survival_hours"],
            "survival_str":    prediction["survival_str"],
            "action":          prediction["action"],
            "pred_confidence": prediction["prediction_conf"],
        },
        "detections": [{"class": d["class_name"], "confidence": d["confidence"],
                        "box": d["box"], "severity_weight": d["severity_weight"]}
                       for d in detections],
    }

    col1, col2 = st.columns(2)
    with col1:
        json_str = json.dumps(report, indent=2)
        st.download_button("⬇ Download JSON Report", json_str,
                           f"hullguard_report_{vessel_name.replace(' ','_')}.json", "application/json")
    with col2:
        if detections:
            df_exp = pd.DataFrame([{
                "Vessel": vessel_name, "Type": vessel_type, "Age": vessel_age,
                "Timestamp": report_time, "Class": d["class_name"],
                "Confidence": d["confidence"], "Severity Weight": d["severity_weight"],
                "Damage %": analysis["damage_pct"], "Risk Level": risk,
                "Survival (hrs)": prediction["survival_hours"],
            } for d in detections])
            st.download_button("⬇ Download CSV Log", df_exp.to_csv(index=False),
                               f"hullguard_{vessel_name.replace(' ','_')}.csv", "text/csv")

    st.json(report)
