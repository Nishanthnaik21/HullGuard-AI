import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
from utils.hull_utils import *

st.set_page_config(page_title="HullGuard — Survival", page_icon="⏱️", layout="wide")
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
</style>""", unsafe_allow_html=True)

nav_header("SURVIVAL PREDICTOR", "RULE-BASED HULL INTEGRITY & SURVIVAL TIME ESTIMATION", "⏱️")

# ── Vessel config sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Orbitron,monospace;font-size:0.85rem;color:#00d4ff;letter-spacing:0.1em;margin-bottom:12px">🚢 VESSEL PARAMETERS</div>', unsafe_allow_html=True)
    vessel_type = st.selectbox("Vessel Type", ["Cargo","Tanker","Passenger","Naval","Fishing","Recreational"])
    vessel_age  = st.slider("Vessel Age (years)", 0, 50, 10)
    sea_state   = st.selectbox("Current Sea State", ["Calm (0–1 m)","Moderate (1–2.5 m)","Rough (2.5–4 m)","Very Rough (4–6 m)","Storm (>6 m)"])
    distance    = st.number_input("Distance to nearest port (nm)", 0, 2000, 50)

    st.markdown("---")
    theory_box("Survival time is estimated using a rule table that maps damage percentage + severity score "
               "to base survival hours. Adjustments are applied for vessel age, type, sea state, and distance. "
               "This is a decision-support tool — always consult a marine surveyor.")

# ── Get data from session or allow manual override ─────────────────────────────
st.markdown("### Input Parameters")

if "analysis" in st.session_state and st.session_state["analysis"]:
    analysis   = st.session_state["analysis"]
    damage_pct = analysis["damage_pct"]
    severity   = analysis["severity_score"]
    st.success(f"✅ Using results from Detection pipeline: Damage={damage_pct:.2f}%, Severity={severity:.1f}")
    use_manual = st.checkbox("Override with manual values")
else:
    st.info("No detection data found — enter values manually below.")
    use_manual = True

if use_manual:
    col1, col2 = st.columns(2)
    with col1:
        damage_pct = st.slider("Damage Area (%)", 0.0, 100.0, 15.0, 0.5)
    with col2:
        severity = st.slider("Severity Score (0–100)", 0.0, 100.0, 35.0, 0.5)

# ── Sea state adjustment ───────────────────────────────────────────────────────
sea_factors = {
    "Calm (0–1 m)":        1.0,
    "Moderate (1–2.5 m)":  0.85,
    "Rough (2.5–4 m)":     0.65,
    "Very Rough (4–6 m)":  0.40,
    "Storm (>6 m)":        0.20,
}
sea_factor = sea_factors[sea_state]

# ── Run prediction ─────────────────────────────────────────────────────────────
prediction = predict_survival(damage_pct, severity, vessel_age, vessel_type)
adjusted_hours = prediction["survival_hours"] * sea_factor

if adjusted_hours >= 168:
    adj_str = f"{adjusted_hours/168:.1f} weeks"
elif adjusted_hours >= 24:
    adj_str = f"{adjusted_hours/24:.1f} days"
elif adjusted_hours >= 1:
    adj_str = f"{adjusted_hours:.1f} hours"
else:
    adj_str = f"{adjusted_hours*60:.0f} minutes"

colour = prediction["colour"]
risk   = prediction["risk_level"]

st.markdown("<br>", unsafe_allow_html=True)

# ── Main result card ───────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{colour}11;border:2px solid {colour};border-radius:12px;
            padding:24px 28px;margin:8px 0 20px 0;position:relative;overflow:hidden">
  <div style="position:absolute;top:0;right:0;width:120px;height:120px;background:{colour}08;
              border-radius:0 12px 0 120px"></div>
  <div style="font-family:Orbitron,monospace;font-size:0.72rem;color:{colour};
              letter-spacing:0.2em;margin-bottom:12px">HULL INTEGRITY VERDICT</div>
  <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap">
    <div>
      <div style="font-family:Orbitron,monospace;font-size:2.2rem;font-weight:800;color:{colour}">
        {adj_str}
      </div>
      <div style="font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#5a7a96;margin-top:4px">
        ESTIMATED SURVIVAL TIME (sea-adjusted)
      </div>
    </div>
    <div style="background:{colour}22;border:1.5px solid {colour};border-radius:8px;
                padding:10px 20px;font-family:Orbitron,monospace;font-size:1rem;
                font-weight:700;color:{colour};letter-spacing:0.1em">
      ⚠ {risk}
    </div>
  </div>
  <div style="font-family:Share Tech Mono,monospace;font-size:0.82rem;color:#c8dff0;
              margin-top:16px;padding-top:16px;border-top:1px solid {colour}44">
    ACTION: {prediction["action"]}
  </div>
</div>""", unsafe_allow_html=True)

# ── Detail metrics ─────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Base Survival", prediction["survival_str"])
c2.metric("Sea Adjusted",  adj_str)
c3.metric("Age Factor",    f"×{prediction['age_factor']:.2f}")
c4.metric("Vessel Factor", f"×{prediction['type_factor']:.2f}")
c5.metric("Sea Factor",    f"×{sea_factor:.2f}")
c6.metric("Pred. Confidence", f"{prediction['prediction_conf']}%")

# ── Port reachability ──────────────────────────────────────────────────────────
if distance > 0:
    speed_knots  = 12.0  # typical cargo speed
    travel_hours = distance / speed_knots
    can_reach    = travel_hours < adjusted_hours
    reach_colour = "#00ff9d" if can_reach else "#ff2d55"
    reach_text   = "CAN REACH PORT" if can_reach else "CANNOT REACH PORT SAFELY"
    st.markdown(f"""
    <div style="background:{reach_colour}11;border:1px solid {reach_colour};border-radius:8px;
                padding:14px 18px;margin:16px 0;display:flex;justify-content:space-between;align-items:center">
      <div>
        <div style="font-family:Orbitron,monospace;font-size:0.78rem;color:{reach_colour};letter-spacing:0.1em">
          {reach_text}
        </div>
        <div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#5a7a96;margin-top:4px">
          {distance} nm at {speed_knots} kn = {travel_hours:.1f} hrs travel | Survival: {adjusted_hours:.1f} hrs
        </div>
      </div>
      <div style="font-size:1.6rem">{'✅' if can_reach else '🆘'}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈  SURVIVAL CURVES", "📋  RULE TABLE", "🔬  METHODOLOGY"])

with tab1:
    # Survival time vs damage % curves for different vessel types
    dmg_range = np.linspace(0, 100, 200)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#050d1a")
    for ax in axes: ax.set_facecolor("#0d1f35")

    v_colours = {"Cargo":"#00d4ff","Tanker":"#ff6b35","Passenger":"#ff2d55","Naval":"#00ff9d"}
    for vtype, vc in v_colours.items():
        hours = [predict_survival(d, d*0.6, vessel_age, vtype)["survival_hours"] for d in dmg_range]
        axes[0].plot(dmg_range, hours, color=vc, linewidth=2, label=vtype)
    axes[0].axvline(damage_pct, color="#ffdd00", linestyle="--", alpha=0.8, label=f"Current ({damage_pct:.1f}%)")
    axes[0].axhline(adjusted_hours, color=colour, linestyle=":", alpha=0.8)
    axes[0].set_title("Survival Time vs Damage %", color="#c8dff0", fontsize=10)
    axes[0].set_xlabel("Damage %", color="#5a7a96"); axes[0].set_ylabel("Hours", color="#5a7a96")
    axes[0].legend(fontsize=8); axes[0].tick_params(colors="#5a7a96")
    for sp in axes[0].spines.values(): sp.set_color("#1a3a5c")

    # Survival vs vessel age
    ages = range(0, 51, 5)
    age_hours = [predict_survival(damage_pct, severity, a, vessel_type)["survival_hours"] * sea_factor for a in ages]
    axes[1].plot(ages, age_hours, color="#00d4ff", linewidth=2, marker="o", markersize=5)
    axes[1].axvline(vessel_age, color="#ffdd00", linestyle="--", alpha=0.8, label=f"Current age ({vessel_age}y)")
    axes[1].set_title(f"Survival vs Vessel Age ({vessel_type})", color="#c8dff0", fontsize=10)
    axes[1].set_xlabel("Vessel Age (years)", color="#5a7a96"); axes[1].set_ylabel("Hours", color="#5a7a96")
    axes[1].legend(fontsize=8); axes[1].tick_params(colors="#5a7a96")
    for sp in axes[1].spines.values(): sp.set_color("#1a3a5c")

    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); buf.seek(0)
    st.image(buf, use_container_width=True); plt.close(fig)

with tab2:
    st.markdown("**Survival prediction rule table:**")
    headers = ["Severity range","Damage %","Base survival","Risk level","Action"]
    rule_rows = []
    for rule in SURVIVAL_RULES:
        min_sev, max_sev, min_dmg, max_dmg, hours, level, action = rule
        clr = RISK_COLOURS[level]
        if hours >= 168: ht = f"{hours//168}w"
        elif hours >= 24: ht = f"{hours//24}d"
        else: ht = f"{hours}h"
        rule_rows.append((f"{min_sev}–{max_sev}", f"{min_dmg}–{max_dmg}%", ht, level, action, clr))

    for row in rule_rows:
        min_sev, dmg_r, ht, level, action, clr = row
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1.5fr 3fr;gap:8px;
                    background:#0d1f35;border-left:3px solid {clr};border-radius:0 6px 6px 0;
                    padding:8px 14px;margin-bottom:6px;font-size:0.8rem">
          <span style="color:#5a7a96">{min_sev}</span>
          <span style="color:#c8dff0">{dmg_r}</span>
          <span style="font-family:Share Tech Mono,monospace;color:{clr}">{ht}</span>
          <span style="color:{clr};font-weight:600">{level}</span>
          <span style="color:#5a7a96">{action}</span>
        </div>""", unsafe_allow_html=True)

with tab3:
    theory_box("Prediction engine uses a rule-based decision table (not ML) for transparency and auditability. "
               "Base survival time is selected by matching current severity score AND damage percentage to the "
               "highest matching rule tier. Age factor = max(0.4, 1.0 − (age−5)×0.03). "
               "Vessel type factors: Cargo=1.0, Tanker=0.85, Passenger=0.70, Naval=1.20. "
               "Sea state multipliers range from 1.0 (calm) to 0.20 (storm). "
               "Port reachability assumes 12-knot average speed.")

    st.markdown("""
    <div style="background:#0d1f35;border:1px solid #1a3a5c;border-radius:8px;padding:16px;
                font-family:Share Tech Mono,monospace;font-size:0.77rem;color:#5a7a96;line-height:2">
    SURVIVAL_ADJ = BASE × age_factor × vessel_factor × sea_factor<br>
    AGE_FACTOR   = max(0.40, 1.0 − (age − 5) × 0.03)<br>
    VESSEL_CARGO=1.0 | TANKER=0.85 | PASSENGER=0.70 | NAVAL=1.20<br>
    SEA_FACTORS: Calm=1.0, Moderate=0.85, Rough=0.65, V.Rough=0.40, Storm=0.20
    </div>""", unsafe_allow_html=True)
