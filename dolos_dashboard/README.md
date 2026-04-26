# DeceptiLens — Deception Detection Dashboard

Predict truth/deception from video clips using your trained multimodal ensemble.

## Project Structure

```
deception_dashboard/
├── app.py                          ← Streamlit dashboard
├── inference.py                    ← Feature extraction + prediction
├── requirements.txt
├── 1_export_models_in_colab.py     ← Run this in your Colab notebook first
├── face_landmarker.task            ← Auto-downloaded on first run
└── saved_models/                   ← Created after running export step
    ├── pytorch_model.pth
    ├── hgb_model.joblib
    ├── svm_model.joblib
    ├── rf_model.joblib
    ├── scaler.joblib
    ├── feature_names.json
    └── model_meta.json
```

---

##  Step-by-Step Deployment

### Step 1 — Export Models from Colab

Open your Colab notebook. After training finishes (after Cell 5),
add a new cell and paste the contents of `1_export_models_in_colab.py`.
Run it. It will save models to:
`/content/drive/MyDrive/Deception_Capstone/saved_models/`

Download that entire `saved_models/` folder to your local machine.

---

### Step 2 — Local Setup

```bash
# 1. Clone / create your project folder
mkdir deception_dashboard && cd deception_dashboard

# 2. Copy all files here (app.py, inference.py, requirements.txt)
# 3. Copy your downloaded saved_models/ folder here

# 4. Create a virtual environment
python -m venv venv
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt
```

---

### Step 3 — Run the Dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.
Upload any .mp4 clip and get instant predictions!

## Model Performance

| Model                    | Accuracy  |
|--------------------------|-----------|
| PyTorch Dual-Stream      | **82.11%** |
| SVM                      | 79.82%    |
| HistGradientBoosting     | 74.77%    |
| Hard Voting Ensemble     | 79.82%    |
| 10-Fold CV Mean          | 81.27%    |

The dashboard uses the **PyTorch Dual-Stream** as the primary verdict
and shows all 4 model scores for transparency.

---

## Inference Time
- First run: ~60s (downloads face_landmarker.task ~30MB)
- Subsequent runs: ~15–40s depending on video length and hardware
- GPU: 3–10s

## Troubleshooting

**`saved_models/ not found`** — Run the Colab export cell first and download the folder.

**`mediapipe` install error** — Try: `pip install mediapipe==0.10.9`

**CUDA out of memory** — Inference falls back to CPU automatically.

**Face not detected** — MediaPipe features default to 0.0; audio + ResNet still run.
