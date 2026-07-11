# HullGuard AI — Marine Hull Damage Detection System
### Full implementation for course project 22AIM61 — DIP

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```
Open: `http://localhost:8501`

---

## Project Structure

```
hull_damage_system/
├── app.py                     ← Home + system overview
├── requirements.txt
├── utils/
│   └── hull_utils.py          ← ALL shared logic:
│                                   preprocess_image()
│                                   run_mock_detection()      ← swap with real YOLO
│                                   run_yolo_detection()      ← real YOLO hook (commented)
│                                   calculate_damage_percentage()
│                                   generate_heatmap()
│                                   predict_survival()
│                                   SURVIVAL_RULES table
└── pages/
    ├── 1_Detection.py         ← Upload + YOLO detection
    ├── 2_Analysis.py          ← Pixel analysis + heatmap
    ├── 3_Survival.py          ← Survival time prediction
    └── 4_Dashboard.py         ← Full report + export
```

---

## Step-by-Step Implementation Guide

### STEP 1 — Input Module (app.py + page 1)
- User uploads hull image via `st.file_uploader`
- Supports JPG, PNG, TIFF
- Image loaded as RGB NumPy array using PIL
- Stored in `st.session_state` for cross-page access

### STEP 2 — Preprocessing Module (utils/hull_utils.py → preprocess_image())
Operations in order:
1. `cv2.resize(img, (640,640))` — standard YOLO input size
2. `cv2.cvtColor(RGB→GRAY)` — grayscale conversion
3. `cv2.fastNlMeansDenoising(h=10)` — sensor noise removal
4. `cv2.createCLAHE(clipLimit=3.0)` or `equalizeHist` — contrast enhancement
5. `cv2.Canny(50,150)` — edge detection
6. `cv2.morphologyEx(MORPH_CLOSE, 3×3, iter=2)` — gap closing

### STEP 3 — AI Detection Module (utils/hull_utils.py → run_mock_detection / run_yolo_detection)

#### Running with mock detection (no GPU needed):
- Mock generates realistic bounding boxes seeded from image content
- Returns: class_name, confidence, box [x1,y1,x2,y2], severity_weight

#### Integrating real YOLOv8:
```python
from ultralytics import YOLO
model = YOLO("best.pt")              # your trained weights
results = model(img_array, conf=0.35, iou=0.45)[0]
for box in results.boxes:
    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
    cls  = int(box.cls[0])
    conf = float(box.conf[0])
    # map to DAMAGE_CLASSES dict
```

#### Training your own YOLO model:
```bash
pip install ultralytics
yolo train model=yolov8n.pt data=hull_damage.yaml epochs=100 imgsz=640
```
Datasets: COCO-format annotations, Hull Defect Dataset (Kaggle), custom photos

### STEP 4 — Damage Analysis Module (utils/hull_utils.py → calculate_damage_percentage())
Algorithm:
1. Create binary damage_mask (640×640, uint8)
2. Fill each detection bounding box in mask with 255
3. `damage_pct = (mask>0).sum() / total_pixels * 100`
4. Per-class stats: area%, max_confidence, edge_density (Canny within ROI)
5. Severity score = mean(confidence × severity_weight) × 100

Severity weights:
- Surface Crack:   0.20
- Corrosion Patch: 0.35
- Deformation:     0.45
- Deep Fracture:   0.55
- Breach / Hole:   0.90

### STEP 5 — Prediction Module (utils/hull_utils.py → predict_survival())
Rule-based lookup table (7 tiers):
```
Severity   Damage%   Base survival   Risk level
0–15       0–5%      720 hours       MINIMAL
0–25       0–10%     480 hours       LOW
15–40      5–15%     168 hours       MODERATE
30–55      10–25%    72 hours        ELEVATED
45–70      20–40%    24 hours        HIGH
60–85      35–60%    8 hours         CRITICAL
75–100     50–100%   2 hours         EXTREME
```

Adjustment multipliers:
- Age: `max(0.4, 1.0 - (age-5) * 0.03)`
- Vessel type: Cargo=1.0, Tanker=0.85, Passenger=0.70, Naval=1.20
- Sea state: Calm=1.0, Moderate=0.85, Rough=0.65, Very Rough=0.40, Storm=0.20

### STEP 6 — Web Dashboard (pages/4_Dashboard.py)
Features:
- Full detection result + heatmap side by side
- 6-panel analytics chart (Matplotlib)
- Detection log as Pandas DataFrame
- Export as JSON or CSV

### STEP 7 — Real-time Monitoring Extension
For real-time webcam/RTSP feed, add:
```python
import streamlit as st
import cv2

# In page file:
cap = cv2.VideoCapture("rtsp://camera_ip/stream")
frame_placeholder = st.empty()
while True:
    ret, frame = cap.read()
    processed = preprocess_image(frame)
    detections = run_yolo_detection(processed["resized"], "best.pt", 0.35, 0.45)
    annotated = draw_detections(processed["resized"], detections)
    frame_placeholder.image(annotated)
```

---

## DIP Modules Coverage

| Module | Operations Used |
|--------|----------------|
| M1 — Fundamentals | Pixel analysis, intensity histograms, image statistics |
| M2 — Spatial/Freq | CLAHE, Gaussian blur, Canny edges, histogram equalisation |
| M3 — Restoration | FastNLM denoising, Wiener-concept noise reduction |
| M4 — Morphology | Morphological close (MORPH_CLOSE), kernel structuring |
| M5 — Segmentation | Canny edge detection, contour extraction, thresholding, bounding box detection |

---

## How to Replace Mock with Real YOLO

1. Collect hull damage images (minimum 500 per class)
2. Annotate using Roboflow or LabelImg → YOLO format
3. Create `hull_damage.yaml`:
   ```yaml
   path: ./dataset
   train: images/train
   val: images/val
   nc: 5
   names: [Surface Crack, Deep Fracture, Corrosion Patch, Deformation, Breach/Hole]
   ```
4. Train: `yolo train model=yolov8n.pt data=hull_damage.yaml epochs=100`
5. In `hull_utils.py`, uncomment `run_yolo_detection()` body and set `model_path="runs/detect/train/weights/best.pt"`

---

Built with Python · OpenCV · YOLOv8 · Streamlit · Matplotlib · Pandas
Course 22AIM61 — Digital Image Processing
