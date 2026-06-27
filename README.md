# Crop Yield Prediction — Project Overview

An end-to-end machine learning research project that forecasts agricultural crop
yield (`hg/ha`) across 101 countries and 10 crop types (1990–2013), using
agro-climatic features (rainfall, pesticide use, temperature, historical yields).

The project is organized as three successive model generations, each improving on
the last in architecture or problem framing.

---

## Repository Layout

```
cropprediction/
├── old_model/          # v1 — baseline: 1D-CNN + Recursive BiLSTM (now-casting)
├── new_model/          # v2 — modular pipeline with hybrid LSTM-XGBoost
├── new_model2/         # v3 — 8-model now-casting suite + interactive dashboard
├── new_model3.0/       # v4 — true one-year-ahead BiLSTM forecaster (current best)
├── crop_yield_pipeline.py   # standalone pipeline script (early prototype)
└── ICRISAT-District Level Data.csv   # district-level India data (secondary dataset)
```

---

## Model Generations

### `old_model` — Baseline CNN + BiLSTM
A full ML pipeline (data analysis → preprocessing → feature selection → training → evaluation) on the global FAO dataset.
Models: Random Forest, Gradient Boosting, 1D-CNN, Recursive BiLSTM, CNN+BiLSTM hybrid.
Includes a Dash-based interactive dashboard.

See [`old_model/README.md`](old_model/README.md).

---

### `new_model` — Modular LSTM-XGBoost Pipeline
Refactored, modular pipeline applied to the ICRISAT district-level (India) dataset.
Introduces LSTM→XGBoost hybrid: LSTM extracts temporal features, XGBoost uses them alongside tabular features.

See the numbered scripts (`01_preprocessing.py` → `04_evaluation.py`) and `main.py`.

---

### `new_model2` — 8-Model Now-Casting Suite
Applies the modular pipeline to the global FAO dataset. Compares 8 model variants
head-to-head, from simple regression to Bidirectional LSTM hybrids.

> **Best result (UniLSTM, test 2011–2013): R² = 0.985 · RMSE = 11,967 hg/ha · 87.9 % within ±20 %**

Includes a glass-morphism Flask + Chart.js web dashboard (`app.py`).

See [`new_model2/README.md`](new_model2/README.md).

---

### `new_model3.0` — True One-Year-Ahead BiLSTM Forecaster
The most architecturally principled model. Unlike `new_model2` (which is given
the target year's own weather — now-casting), this model uses **only past years**
to predict one year ahead — a strictly harder and more practically useful task.

Architecture: 2× Bidirectional LSTM with learned Country + Crop embeddings,
trained on 4-year windows to predict the 5th year's yield.

> **Test result (2011–2013): R² = 0.953 · RMSE = 19,215 hg/ha · 67.9 % within ±20 %**

Includes a full Flask web dashboard with live inference (`app.py`).

See [`new_model3.0/README.md`](new_model3.0/README.md).

---

## Dataset

**Primary:** FAO global crop yield panel (`new_model2/yield.csv`, merged from
`rainfall.csv` and `pesticides.csv`).

| Property   | Value                               |
|------------|-------------------------------------|
| Records    | 28,242 rows                         |
| Countries  | 101                                 |
| Crops      | 10                                  |
| Years      | 1990 – 2013                         |
| Target     | `hg/ha_yield` (hectograms/hectare)  |
| Features   | rainfall (mm/yr), pesticides (t), avg. temperature, lagged yield |

**Secondary:** ICRISAT district-level India data (`ICRISAT-District Level Data.csv`),
used in `new_model/`.

---

## Quickstart (latest model — `new_model3.0`)

```bash
# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r new_model3.0/requirements.txt

# 3. train the model (writes outputs/ artifacts)
cd new_model3.0
python bilstm_yield_model.py

# 4. launch the interactive dashboard
python app.py                      # -> http://127.0.0.1:5000
```

---

## Results Summary

| Model version | Task | Test R² | RMSE (hg/ha) | Within ±20 % |
|---------------|------|---------|--------------|--------------|
| `old_model` — CNN+BiLSTM | now-cast | ~0.93 | — | — |
| `new_model2` — UniLSTM (best) | now-cast | **0.985** | 11,967 | 87.9 % |
| `new_model2` — BiLSTM | now-cast | 0.984 | — | 82.0 % |
| `new_model3.0` — BiLSTM | **1-yr forecast** | **0.953** | 19,215 | 67.9 % |

`new_model3.0` solves the harder problem (no future data leakage) and is the
recommended model for real-world forward-looking use.

---

## Requirements

See `new_model3.0/requirements.txt` for the latest model. Core dependencies:

- Python 3.10+
- TensorFlow ≥ 2.16
- scikit-learn ≥ 1.3
- pandas, numpy, matplotlib
- Flask ≥ 3.0
