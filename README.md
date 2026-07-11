# HullGuard AI — Marine Hull Damage Detection System

### Full implementation for course project 22AIM61 — Digital Image Processing (DIP)

A web-based AI system for detecting hull damage, calculating damage severity, and predicting vessel survival time — built with YOLOv8, OpenCV, and Streamlit.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
streamlit run app.py

# 3. Access the web dashboard
# Open your browser and go to: http://localhost:8501
```

---

## Project Structure 

```text
hull_damage_system/
├── app.py                          ← Home dashboard & system overview
├── requirements.txt
├── utils/
│   └── hull_utils.py               ← Core pipeline (preprocessing, detection, analysis, prediction)
└── pages/
    ├── 1_Detection.py              ← Main analysis dashboard & file upload
    ├── 2_Analysis.py               ← Step-by-step preprocessing & pixel analysis visualiser
    ├── 3_Survival.py               ← Survival time prediction breakdown
    └── 4_Dashboard.py              ← Full metrics report + JSON/CSV export
```

---

## Step-by-Step Implementation & Module Breakdown

### 1. Input Module (`app.py` & `pages/1_Detection.py`)
- User uploads hull image via Streamlit (JPG, PNG, TIFF supported).
- Demo synthetic hull image available if no upload provided.
- Settings sidebar: confidence threshold, sea state, vessel age, material.

### 2. Preprocessing Module (`utils/hull_utils.py → preprocess_image()`)
| Step | Operation | Purpose / DIP Concept |
|------|-----------|---------|
| 1 | Resize 640×640 | Standardize YOLO input size |
| 2 | Grayscale Conversion | Base format for structural analysis |
| 3 | FastNLMeansDenoising | Sensor noise reduction (M3 - Restoration) |
| 4 | CLAHE (Contrast Normalization) | Contrast enhancement (M2 - Spatial/Freq) |
| 5 | Canny Edge Detection | Edge identification (M5 - Segmentation) |
| 6 | Morphological Close (3x3) | Gap closing (M4 - Morphology) |

### 3. AI Detection Module (`utils/hull_utils.py → run_yolo_detection()`)
- **Stub/Mock Mode** (no GPU needed): Intelligent edge/blob analysis simulates YOLO output for testing.
- **Production Mode**: Integrate trained `best.pt` YOLOv8 weights. Returns: class, confidence, bounding box coordinates, pixel area per detection.

### 4. Damage Analysis Module (`utils/hull_utils.py → calculate_damage_percentage()`)
- Algorithm generates a binary `damage_mask` (640x640) from all detection bounding boxes.
- Calculates Raw Damage % = (damaged pixels / total pixels) * 100.
- Applies severity-weighted damage based on class (e.g., Hole=0.90, Deformation=0.45, Crack=0.20).
- Calculates Detection Accuracy = Mean YOLO confidence across all detections.

### 5. Prediction Module (`utils/hull_utils.py → predict_survival()`)
Rule-based lookup table predicting vessel survival time based on damage and environmental modifiers:

| Weighted Damage | Risk Level | Base Survival Time |
|-----------------|------------|---------------|
| 0–5%            | SAFE / MINIMAL | 10–30 days (720 hours) |
| 5–15%           | MONITOR / MODERATE | 3–10 days (168 hours) |
| 15–40%          | ELEVATED / HIGH | 24–72 hours   |
| >40%            | CRITICAL / ABANDON | < 8 hours     |

**Environmental Modifiers Applied:**
- **Sea state** (Calm to Storm): Alters time by ×1.0 down to ×0.20
- **Vessel age**: Decreases structural integrity multiplier based on years active
- **Hull material / Vessel Type**: Cargo=×1.0, Tanker=×0.85, Passenger=×0.70, Naval=×1.20

### 6. Web Dashboard (`pages/4_Dashboard.py`)
- Annotated detection image alongside severity heatmap.
- Detection log table with class, confidence, and coordinates (Pandas DataFrame).
- 6-panel analytics chart (Matplotlib) reporting metrics and severity score.
- Full JSON report or CSV export of damage statistics and risk assessment.

---

## Output Metrics Explained

| Metric | How Calculated |
|--------|---------------|
| **Hull Damage %** | Damaged pixels (union mask) / total pixels × 100 |
| **Weighted Damage %** | Damage % × severity_score (type-weighted) |
| **Detection Accuracy** | Mean YOLO confidence of all positive detections |
| **Severity Score** | Σ(area × type_weight × confidence) / Σ(area) |
| **Survival Time** | Rule table lookup on effective damage + env modifiers |
| **Risk Level** | Multi-band classification (SAFE to EXTREME/ABANDON) |

---

## Training Your Own Model (Replacing Mock with Real YOLO)

1. Collect hull damage images (minimum 500 per class from sources like Kaggle or Roboflow Universe).
2. Annotate images in YOLO format.
3. Create a `hull_damage.yaml` file:
   ```yaml
   path: ./dataset
   train: images/train
   val: images/val
   nc: 5
   names: [Surface Crack, Deep Fracture, Corrosion Patch, Deformation, Breach/Hole]
   ```
4. Train the model:
   ```bash
   pip install ultralytics
   yolo train model=yolov8n.pt data=hull_damage.yaml epochs=100 imgsz=640
   ```
5. In `hull_utils.py`, uncomment the `run_yolo_detection()` body and set `model_path` to your resulting `best.pt` weights.

---

---

*Built with: Python · OpenCV · YOLOv8 (Ultralytics) · Streamlit · PyTorch · NumPy · Pandas · Matplotlib*
