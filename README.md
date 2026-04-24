# 🍅 TomatoScan AI — Tomato Leaf Disease Detection

> **VGG16 Transfer Learning · PlantVillage Dataset · Q1 Journal Ready**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com)

---

## 📁 Project Structure

```
tomato-disease-project/
│
├── notebook/
│   └── tomato_disease_vgg16_research.py   ← Full training pipeline (run on Kaggle/local)
│
├── backend/
│   ├── app.py                             ← Flask REST API
│   ├── requirements.txt
│   ├── Dockerfile
│   └── models/                            ← Place trained model files here
│       ├── tomato_disease_vgg16.h5
│       └── class_labels.json
│
├── frontend/
│   └── index.html                         ← Complete single-page web application
│
├── docker-compose.yml                     ← Full-stack deployment
├── nginx.conf
└── README.md
```

---

## 🚀 Quick Start

### Step 1: Train the Model (Kaggle Recommended)

1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook
2. Add dataset: **Settings → Add data → search `kaustubhb999/tomatoleaf`**
3. Enable **GPU accelerator** (Settings → Accelerator → GPU P100)
4. Upload `notebook/tomato_disease_vgg16_research.py` and run all cells
5. Download from `/kaggle/working/models/`:
   - `tomato_disease_vgg16.h5`
   - `class_labels.json`

### Step 2: Set Up Backend

```bash
cd backend
mkdir -p models
# Copy tomato_disease_vgg16.h5 and class_labels.json into models/

pip install -r requirements.txt
python app.py
# API runs at http://localhost:5000
```

### Step 3: Open Frontend

Simply open `frontend/index.html` in your browser.
The page connects to `http://localhost:5000` by default.

> **To change API URL:** Edit `const API_BASE = 'http://localhost:5000';` in `index.html`

---

## 🐳 Docker Deployment (Production)

```bash
# 1. Copy trained model files
cp path/to/tomato_disease_vgg16.h5   backend/models/
cp path/to/class_labels.json         backend/models/

# 2. Build and start
docker-compose up --build -d

# Frontend: http://localhost
# Backend:  http://localhost:5000
```

---

## 🔌 API Reference

### `GET /health`
Returns model status.
```json
{ "status": "ok", "model_loaded": true, "num_classes": 10, "tensorflow": "2.16.2" }
```

### `GET /classes`
Returns all 10 class names with disease info.

### `POST /predict`
Upload a tomato leaf image.

**Request:** `multipart/form-data` with field `image`

**Response:**
```json
{
  "is_tomato": true,
  "top_prediction": {
    "class": "Tomato___Early_blight",
    "short": "Early Blight",
    "confidence": 94.7,
    "info": {
      "description": "...",
      "treatment": "...",
      "severity": "Moderate"
    }
  },
  "top3": [...],
  "thumb_base64": "...",
  "latency_ms": 142
}
```

### `POST /gradcam`
Returns a Grad-CAM overlay image as base64.

**Request:** `multipart/form-data` with field `image`

**Response:**
```json
{ "gradcam_base64": "..." }
```

---

## 🏗️ Model Architecture

```
Input (224×224×3)
     │
VGG16 Backbone (ImageNet weights)
  ├── block1–block4  [FROZEN in both phases]
  └── block5         [FROZEN in Phase 1, UNFROZEN in Phase 2]
     │
GlobalAveragePooling2D → (512,)
     │
Dense(512) → BatchNorm → ReLU → Dropout(0.5)
     │
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
     │
Dense(10, softmax)
```

### Training Strategy

| Phase | Epochs | LR     | What's trained          |
|-------|--------|--------|-------------------------|
| 1     | 15     | 1e-4   | Head only (VGG16 frozen)|
| 2     | 20     | 1e-5   | Head + VGG16 block5     |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source   | [kaustubhb999/tomatoleaf (Kaggle)](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf) |
| Classes  | 10 (9 diseases + Healthy) |
| Split    | 70% train / 15% val / 15% test (stratified) |
| Augmentation | Rotation±40°, shift, shear, zoom, flip, brightness [0.8–1.2] |

---

## 📈 Expected Performance (published benchmarks)

| Metric         | Expected   |
|----------------|------------|
| Test Accuracy  | ≥ 97%      |
| Macro F1       | ≥ 0.96     |
| AUC-ROC        | ≥ 0.99     |
| Cohen's Kappa  | ≥ 0.96     |
| MCC            | ≥ 0.96     |

---

## 📄 Citation

If you use this work in a publication, please cite:

```bibtex
@article{tomatoscan2024,
  title   = {Tomato Leaf Disease Detection Using VGG16 Two-Phase Transfer Learning},
  author  = {Your Name},
  journal = {Target Journal},
  year    = {2024},
  dataset = {PlantVillage Tomato Leaf Disease Dataset},
  url     = {https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf}
}
```

---

## 🌿 Supported Disease Classes

| # | Class | Description |
|---|-------|-------------|
| 0 | Bacterial Spot | Dark lesions caused by Xanthomonas |
| 1 | Early Blight | Concentric ring lesions (Alternaria solani) |
| 2 | Late Blight | Water-soaked spots (Phytophthora infestans) |
| 3 | Leaf Mold | Yellow upper spots, olive-grey lower mold |
| 4 | Septoria Leaf Spot | Small circular spots (Septoria lycopersici) |
| 5 | Spider Mites | Stippled leaves with fine webbing |
| 6 | Target Spot | Concentric rings (Corynespora cassiicola) |
| 7 | Yellow Leaf Curl Virus | Viral, whitefly-transmitted |
| 8 | Tomato Mosaic Virus | Mosaic pattern, no cure |
| 9 | Healthy | No disease detected |
