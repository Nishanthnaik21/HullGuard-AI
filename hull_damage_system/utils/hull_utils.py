"""
utils/hull_utils.py
Shared utilities for HullGuard AI:
  - Image preprocessing pipeline
  - Mock YOLO detection (runs without GPU/model file for demo)
  - Real YOLO integration hooks
  - Damage percentage calculation
  - Survival time prediction engine
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io, random, math

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DAMAGE_CLASSES = {
    0: "Surface Crack",
    1: "Deep Fracture",
    2: "Corrosion Patch",
    3: "Deformation",
    4: "Breach / Hole",
}

CLASS_SEVERITY = {
    "Surface Crack":   0.20,
    "Deep Fracture":   0.55,
    "Corrosion Patch": 0.35,
    "Deformation":     0.45,
    "Breach / Hole":   0.90,
}

CLASS_COLOURS_BGR = {
    "Surface Crack":   (0, 200, 255),
    "Deep Fracture":   (0, 80, 255),
    "Corrosion Patch": (0, 180, 120),
    "Deformation":     (0, 165, 255),
    "Breach / Hole":   (0, 40, 220),
}

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING MODULE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(img_array: np.ndarray, mode: str = "Standard") -> dict:
    """
    Full preprocessing pipeline.
    Returns dict with all intermediate images.
    """
    # 1. Resize to standard inference size
    resized = cv2.resize(img_array, (640, 640))

    # 2. Convert to grayscale for analysis
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    # 3. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 4. Enhance contrast
    if mode == "Enhanced (CLAHE)":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
    elif mode == "Edge-boosted":
        enhanced = cv2.equalizeHist(denoised)
    else:
        enhanced = denoised

    # 5. Edge detection (Canny)
    edges = cv2.Canny(enhanced, threshold1=50, threshold2=150)

    # 6. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 7. Prepare RGB enhanced version for overlay
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return {
        "original":     img_array,
        "resized":      resized,
        "gray":         gray,
        "denoised":     denoised,
        "enhanced":     enhanced,
        "enhanced_rgb": enhanced_rgb,
        "edges":        edges,
        "morph":        morph,
    }

# ─────────────────────────────────────────────────────────────────────────────
# AI DETECTION MODULE  (Mock + Real hooks)
# ─────────────────────────────────────────────────────────────────────────────

def run_mock_detection(img_array: np.ndarray, conf_threshold: float = 0.35) -> list:
    """
    Mock YOLO detection — generates realistic bounding boxes and labels.
    Replace this function body with real YOLO call once model is available.
    
    Returns list of dicts:
        {class_name, confidence, box:[x1,y1,x2,y2], severity_weight}
    """
    h, w = img_array.shape[:2]
    detections = []

    # Seed based on image content for reproducibility
    seed_val = int(img_array.mean() * 1000) % 10000
    rng = random.Random(seed_val)

    # Number of detections varies with image brightness (simulates damage)
    brightness = img_array.mean()
    n_detections = rng.randint(2, 6) if brightness < 128 else rng.randint(1, 4)

    used_areas = []
    for _ in range(n_detections):
        # Random box
        for attempt in range(20):
            bw = rng.randint(w // 8, w // 3)
            bh = rng.randint(h // 8, h // 3)
            bx = rng.randint(0, w - bw)
            by = rng.randint(0, h - bh)

            # Avoid heavy overlap
            overlap = False
            for (ox, oy, ow, oh) in used_areas:
                ix = max(0, min(bx+bw, ox+ow) - max(bx, ox))
                iy = max(0, min(by+bh, oy+oh) - max(by, oy))
                if ix * iy > 0.4 * bw * bh:
                    overlap = True; break
            if not overlap:
                used_areas.append((bx, by, bw, bh))
                break

        class_name = rng.choice(list(DAMAGE_CLASSES.values()))
        confidence = round(rng.uniform(conf_threshold + 0.05, 0.97), 3)

        detections.append({
            "class_name":     class_name,
            "confidence":     confidence,
            "box":            [bx, by, bx + bw, by + bh],
            "severity_weight": CLASS_SEVERITY[class_name],
        })

    return detections


def run_yolo_detection(img_array: np.ndarray, model_path: str, conf: float, iou: float) -> list:
    """
    Real YOLOv8 detection hook.
    Uncomment and use when model weights are available.
    
    Usage:
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model(img_array, conf=conf, iou=iou)
        ...
    """
    # ── REAL IMPLEMENTATION (uncomment when model ready) ──────────────────────
    # from ultralytics import YOLO
    # model = YOLO(model_path)
    # results = model(img_array, conf=conf, iou=iou)[0]
    # detections = []
    # for box in results.boxes:
    #     x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    #     cls  = int(box.cls[0])
    #     conf = float(box.conf[0])
    #     name = DAMAGE_CLASSES.get(cls, "Unknown")
    #     detections.append({
    #         "class_name":     name,
    #         "confidence":     conf,
    #         "box":            [x1, y1, x2, y2],
    #         "severity_weight": CLASS_SEVERITY.get(name, 0.5),
    #     })
    # return detections
    # ─────────────────────────────────────────────────────────────────────────
    return run_mock_detection(img_array, conf)


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATED IMAGE RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def draw_detections(img_array: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes with labels on the image."""
    annotated = img_array.copy()
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2RGB)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        name = det["class_name"]
        conf = det["confidence"]
        colour_bgr = CLASS_COLOURS_BGR.get(name, (0, 200, 255))
        colour_rgb = (colour_bgr[2], colour_bgr[1], colour_bgr[0])

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour_rgb, 2)

        # Label background
        label = f"{name}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 8, y1), colour_rgb, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Confidence indicator (small circle)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        radius = min((x2-x1), (y2-y1)) // 8
        cv2.circle(annotated, (cx, cy), max(3, radius), colour_rgb, 1)

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# DAMAGE ANALYSIS MODULE
# ─────────────────────────────────────────────────────────────────────────────

def calculate_damage_percentage(img_array: np.ndarray, detections: list) -> dict:
    """
    Calculate damage percentage using pixel analysis within detected boxes.
    Also computes per-class breakdown and overall severity score.
    """
    h, w = img_array.shape[:2]
    total_pixels = h * w
    gray = img_array if len(img_array.shape) == 2 else cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    damage_mask   = np.zeros((h, w), dtype=np.uint8)
    class_stats   = {}
    box_areas     = []

    for det in detections:
        x1, y1, x2, y2 = [max(0, v) for v in det["box"]]
        x2, y2 = min(x2, w), min(y2, h)
        name = det["class_name"]
        conf = det["confidence"]

        # Mark region in damage mask
        damage_mask[y1:y2, x1:x2] = 255

        # Pixel analysis within bounding box
        roi     = gray[y1:y2, x1:x2]
        roi_pix = roi.size
        box_area_pct = roi_pix / total_pixels * 100

        # Edge density within ROI = proxy for crack severity
        edges_roi  = cv2.Canny(roi, 30, 100)
        edge_pct   = (edges_roi > 0).sum() / max(roi_pix, 1) * 100

        box_areas.append(box_area_pct)

        if name not in class_stats:
            class_stats[name] = {"count": 0, "area_pct": 0.0, "max_conf": 0.0, "edge_density": 0.0}
        class_stats[name]["count"]       += 1
        class_stats[name]["area_pct"]    += box_area_pct
        class_stats[name]["max_conf"]     = max(class_stats[name]["max_conf"], conf)
        class_stats[name]["edge_density"] = max(class_stats[name]["edge_density"], edge_pct)

    # Overall damage area (union of all boxes)
    total_damage_pct = (damage_mask > 0).sum() / total_pixels * 100

    # Weighted severity score (0–100)
    severity_score = 0.0
    if detections:
        for det in detections:
            severity_score += det["confidence"] * det["severity_weight"] * 100
        severity_score = min(severity_score / len(detections), 100)

    # Detection confidence (mean of all box confidences)
    if detections:
        mean_conf = sum(d["confidence"] for d in detections) / len(detections)
    else:
        mean_conf = 0.0

    return {
        "damage_pct":     round(total_damage_pct, 2),
        "severity_score": round(severity_score, 1),
        "mean_confidence": round(mean_conf * 100, 1),
        "num_detections": len(detections),
        "class_stats":    class_stats,
        "damage_mask":    damage_mask,
        "box_areas":      box_areas,
    }


def generate_heatmap(img_array: np.ndarray, damage_mask: np.ndarray) -> np.ndarray:
    """Overlay a pseudo-colour damage heatmap on the original image."""
    if len(img_array.shape) == 2:
        base = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        base = img_array.copy()

    # Blur the damage mask for smooth heatmap
    blurred = cv2.GaussianBlur(damage_mask, (51, 51), 0)
    heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    alpha = 0.45
    overlay = cv2.addWeighted(base, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# SURVIVAL PREDICTION MODULE
# ─────────────────────────────────────────────────────────────────────────────

SURVIVAL_RULES = [
    # (min_severity, max_severity, min_damage%, max_damage%, survival_hours, risk_level, action)
    (0,   15,  0,   5,   720,  "MINIMAL",   "Routine maintenance recommended"),
    (0,   25,  0,   10,  480,  "LOW",       "Schedule inspection within 30 days"),
    (15,  40,  5,   15,  168,  "MODERATE",  "Inspection required within 7 days"),
    (30,  55,  10,  25,  72,   "ELEVATED",  "Immediate inspection — do not sail beyond coastal"),
    (45,  70,  20,  40,  24,   "HIGH",      "Return to port immediately — sailing risk"),
    (60,  85,  35,  60,  8,    "CRITICAL",  "Emergency protocol — SOS ready — abandon if offshore"),
    (75,  100, 50,  100, 2,    "EXTREME",   "ABANDON VESSEL — hull failure imminent"),
]

RISK_COLOURS = {
    "MINIMAL":  "#00ff9d",
    "LOW":      "#88ff44",
    "MODERATE": "#ffdd00",
    "ELEVATED": "#ffaa00",
    "HIGH":     "#ff6b35",
    "CRITICAL": "#ff2d55",
    "EXTREME":  "#cc0033",
}

def predict_survival(damage_pct: float, severity_score: float,
                     vessel_age: int = 10, vessel_type: str = "Cargo") -> dict:
    """
    Rule-based survival time prediction.
    
    Factors:
    - Damage percentage (primary)
    - Severity score (weighted by damage class)
    - Vessel age (older = shorter survival)
    - Vessel type (tanker/cargo/passenger have different tolerances)
    """
    # Age multiplier
    age_factor = max(0.4, 1.0 - (vessel_age - 5) * 0.03)

    # Vessel type multiplier
    type_factors = {
        "Cargo":         1.0,
        "Tanker":        0.85,
        "Passenger":     0.70,   # Safety margin
        "Naval":         1.20,
        "Fishing":       0.90,
        "Recreational":  0.75,
    }
    type_factor = type_factors.get(vessel_type, 1.0)

    # Match rule
    matched_rule = SURVIVAL_RULES[0]
    for rule in reversed(SURVIVAL_RULES):
        min_sev, max_sev, min_dmg, max_dmg, hours, level, action = rule
        if severity_score >= min_sev and damage_pct >= min_dmg:
            matched_rule = rule
            break

    _, _, _, _, base_hours, risk_level, action = matched_rule
    adjusted_hours = base_hours * age_factor * type_factor

    # Confidence in prediction (based on number of factors)
    prediction_conf = min(95, 70 + (10 if damage_pct > 5 else 0) + (10 if severity_score > 20 else 0))

    # Format time
    if adjusted_hours >= 168:
        time_str = f"{adjusted_hours/168:.1f} weeks"
    elif adjusted_hours >= 24:
        time_str = f"{adjusted_hours/24:.1f} days"
    elif adjusted_hours >= 1:
        time_str = f"{adjusted_hours:.1f} hours"
    else:
        time_str = f"{adjusted_hours*60:.0f} minutes"

    return {
        "risk_level":     risk_level,
        "survival_hours": round(adjusted_hours, 1),
        "survival_str":   time_str,
        "action":         action,
        "colour":         RISK_COLOURS[risk_level],
        "prediction_conf": prediction_conf,
        "age_factor":     round(age_factor, 2),
        "type_factor":    round(type_factor, 2),
        "base_hours":     base_hours,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def nav_header(title: str, subtitle: str, icon: str):
    st.markdown(f"""
    <div style="padding:16px 0 8px 0; border-bottom:1px solid #1a3a5c; margin-bottom:20px">
      <div style="font-family:'Orbitron',monospace; font-size:1.3rem; font-weight:700;
                  color:#00d4ff; letter-spacing:0.05em">{icon} {title}</div>
      <div style="font-family:'Share Tech Mono',monospace; font-size:0.78rem;
                  color:#5a7a96; letter-spacing:0.12em; margin-top:3px">{subtitle}</div>
    </div>""", unsafe_allow_html=True)


def risk_badge(risk_level: str, large: bool = False):
    colour = RISK_COLOURS.get(risk_level, "#00d4ff")
    size   = "1.1rem" if large else "0.8rem"
    pad    = "10px 20px" if large else "5px 12px"
    st.markdown(f"""
    <div style="display:inline-block; background:{colour}22; border:2px solid {colour};
                border-radius:6px; padding:{pad}; font-family:'Orbitron',monospace;
                font-size:{size}; font-weight:700; color:{colour}; letter-spacing:0.1em">
      ⚠ {risk_level}
    </div>""", unsafe_allow_html=True)


def theory_box(text: str):
    st.markdown(f"""
    <div style="background:#0a1628; border-left:3px solid #00d4ff; border-radius:0 6px 6px 0;
                padding:10px 14px; margin:10px 0; font-size:0.82rem; color:#5a7a96; line-height:1.6">
    📖 {text}
    </div>""", unsafe_allow_html=True)


def load_uploaded_image(uploaded_file) -> np.ndarray | None:
    if uploaded_file is None:
        return None
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)
