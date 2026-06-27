# 🌾 BiLSTM Crop-Yield Prediction — `new_model3.0`

A **Bidirectional LSTM** neural network that forecasts crop yield (`hg/ha`) for a
given **country + crop**, using a multi-year window of agro-climatic history
(yield, rainfall, pesticide use, temperature).

> **Headline result — held-out future test set (2011–2013):**
> **R² = 0.953 · RMSE = 19,215 hg/ha · MAE = 10,810 hg/ha**

---

## 1. Why a BiLSTM?

Crop yield is a **temporal process**: each year's outcome depends on the
trajectory of weather and management over preceding seasons, not just the
current year. A **Bidirectional LSTM** reads each multi-year window in *both*
directions, letting the network relate early-window conditions to late-window
conditions (e.g. how a dry start followed by a wet finish plays out) before
committing to a forecast.

But yield also depends heavily on ***who* and *what*** — a tonne of maize in the
Netherlands is not a tonne of maize in Niger. So identity is modeled separately
via **learned embeddings**, and fused with the temporal signal. This hybrid
design is the core of the architecture.

---

## 2. Dataset

Source: `../new_model2/yield_df.csv` (the FAO district-level panel).

| Property            | Value                                            |
|---------------------|--------------------------------------------------|
| Records             | 28,242 rows                                      |
| Countries (`Area`)  | 101                                              |
| Crops (`Item`)      | 10                                               |
| Years               | 1990 – 2013                                       |
| Target              | `hg/ha_yield` (hectograms per hectare)           |
| Per-year features   | rainfall (mm/yr), pesticides (tonnes), avg. temp |

---

## 3. How the problem is framed

Each `(Country, Crop)` pair is a time series. We slide a **4-year window** over
each series and predict the **5th year's yield**:

```
        ┌──────── input window (T = 4 years) ────────┐   ┌── target ──┐
Year     2003      2004      2005      2006              2007
Yield    y₂₀₀₃     y₂₀₀₄     y₂₀₀₅     y₂₀₀₆     ──────▶  ŷ₂₀₀₇
Rain     r₂₀₀₃     …                                    (predicted)
Pest.    p₂₀₀₃     …
Temp     t₂₀₀₃     …
```

Each timestep carries **4 features** `[yield, rainfall, pesticides, temp]`.
Only **strictly contiguous** year runs are used (no gaps across the window),
which guarantees clean sequences.

### Leakage-safe temporal split

The split is **chronological** (never random) — the model is always asked to
predict the *future*, exactly as it would be used in practice. Scalers are fit
**only on the training period**.

| Split | Target years | Sequences |
|-------|--------------|-----------|
| Train | ≤ 2007       | 4,976     |
| Val   | 2008 – 2010  | 1,733     |
| Test  | 2011 – 2013  | 1,744     |
| **Total** |          | **8,453** |

---

## 4. Model architecture

```
   seq (4×4)                area_id            item_id
       │                       │                  │
 ┌─────▼─────┐          ┌──────▼──────┐    ┌──────▼──────┐
 │ BiLSTM 96 │          │ Embedding   │    │ Embedding   │
 │ (return   │          │ Country→32  │    │ Crop→5      │
 │  seq)     │          └──────┬──────┘    └──────┬──────┘
 └─────┬─────┘                 │                  │
   Dropout 0.25                │                  │
 ┌─────▼─────┐                 │                  │
 │ BiLSTM 48 │                 │                  │
 └─────┬─────┘                 │                  │
   Dropout 0.25                │                  │
       └──────────┬────────────┴──────────────────┘
                  ▼
            Concatenate
                  │
         Dense 128 → BatchNorm → Dropout 0.30
                  │
         Dense 64  → Dropout 0.20
                  │
            Dense 1  →  predicted yield
```

| Component        | Choice                          | Rationale                                            |
|------------------|---------------------------------|------------------------------------------------------|
| Temporal encoder | 2× **Bidirectional LSTM** (96→48) | reads the window forward & backward                  |
| Identity         | **Embeddings** (Country, Crop)  | captures location/crop-specific yield baselines      |
| Regularization   | Dropout + BatchNorm             | prevents overfitting on ~5k training sequences       |
| Loss             | **Huber** (δ=1.0)               | robust to outlier yields vs. plain MSE               |
| Optimizer        | Adam (1e-3) + `ReduceLROnPlateau` | adaptive, with LR annealing                        |
| Stopping         | **EarlyStopping** (patience 15) | restores best validation weights                     |

---

## 5. Results

Metrics are computed on **inverse-scaled** (real-unit) yields.

| Split | R²        | RMSE (hg/ha) | MAE (hg/ha) | MAPE   |
|-------|-----------|--------------|-------------|--------|
| Train | 0.970     | 12,976       | 7,552       | 33.2 % |
| Val   | 0.960     | 17,050       | 9,775       | 26.9 % |
| **Test** | **0.953** | **19,215** | **10,810**  | 27.9 % |

The small train→test gap shows the model **generalizes to unseen future years**
rather than memorizing. (MAPE is inflated by a handful of very-low-yield crop
records where small absolute errors become large percentages — R²/RMSE are the
more reliable indicators here.)

**Accuracy (test set):** 67.9 % of forecasts land within ±20 % of the actual
yield, 79.0 % within ±30 %, and the **median error is just 12 %**.

**Diagnostic plots** (in `outputs/`):
- `training_history.png` — loss & MAE curves (train vs. val)
- `test_predictions.png` — predicted vs. actual scatter on the test set

### Does it beat `new_model2`?

**Honestly — no, not on raw scores, and that is expected.** The two models solve
*different tasks*:

| Model | Task | Test R² | Within ±20 % |
|-------|------|---------|--------------|
| `new_model2` — best (UniLSTM) | now-cast | 0.985 | 87.9 % |
| `new_model2` — BiLSTM | now-cast | 0.984 | 82.0 % |
| **`new_model3.0` — BiLSTM (this)** | **1-yr forecast** | **0.953** | **67.9 %** |

`new_model2` feeds the model the **target year's own** rainfall, pesticide and
temperature values — it explains the current year *given that year's weather*
(**now-casting**). `new_model3.0` is given **only the past** and must predict a
year it has never seen (**true forecasting**) — a strictly harder problem, which
is why its R² is lower yet arguably more useful for planning ahead. Judged as a
forecaster (R² = 0.95 one year out, using no future information), it performs
strongly.

---

## 6. Running it

```bash
# from the repo root, with the project virtualenv active
source .venv/bin/activate
cd new_model3.0
python bilstm_yield_model.py
```

Requires `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`
(see `requirements.txt`). Training takes ~1 min on CPU.

### Artifacts written to `outputs/`

| File                       | Contents                                           |
|----------------------------|----------------------------------------------------|
| `bilstm_yield_model.keras` | trained Keras model                                |
| `preprocessing.pkl`        | label encoders, feature/target scalers, window cfg |
| `metrics.json`             | train/val/test metrics                             |
| `training_history.png`     | learning curves                                    |
| `test_predictions.png`     | predicted-vs-actual scatter                        |

### Reloading for inference

```python
import pickle, numpy as np
from tensorflow import keras

model = keras.models.load_model("outputs/bilstm_yield_model.keras")
prep  = pickle.load(open("outputs/preprocessing.pkl", "rb"))

# Build one window of shape (1, 4, 4): rows = years, cols =
# [hg/ha_yield, rainfall_mm, pesticides_tonnes, avg_temp]
window = np.array([[ ... ]], dtype="float32")          # (1, 4, 4)
window = prep["feature_scaler"].transform(
             window.reshape(-1, 4)).reshape(1, 4, 4)
area = prep["encoders"]["Area"].transform(["India"])
item = prep["encoders"]["Item"].transform(["Maize"])

pred = model.predict({"seq": window, "area": area, "item": item})
yield_hg_ha = prep["target_scaler"].inverse_transform(pred)[0, 0]
```

---

## 7. Interactive web dashboard

A **glass-morphism web dashboard** (Flask + Chart.js + Lucide SVG icons — no
emojis) visualises the findings and lets you **test the model live**.

```bash
source .venv/bin/activate
cd new_model3.0
python app.py            # -> http://127.0.0.1:5000
```

> Train the model first (`python bilstm_yield_model.py`) so `outputs/` exists.
> On startup the app loads the model and **scores all 8,453 sequences** so every
> chart shows real predictions.

**What it shows**

| Panel                       | What it tells you                                              |
|-----------------------------|----------------------------------------------------------------|
| KPI cards                   | headline Test R² / RMSE / MAE / sequence count                 |
| Predicted vs Actual scatter | how tightly test predictions hug the perfect-fit line          |
| Yearly mean yield           | actual vs predicted yield trend across 1994–2013               |
| Accuracy by crop (R²)       | which crops the model nails vs. struggles with                 |
| Residual distribution       | are errors centred & symmetric (no systematic bias)            |
| Top-countries table         | average yield + per-country fit, with inline bars              |
| **Test the Model**          | pick a country + crop, **Autofill** a real historical window, edit any value, and **Predict** — when autofilled, the held-out actual is shown beside the prediction with % error |

**API endpoints** (also usable directly):

| Route            | Method | Purpose                                  |
|------------------|--------|------------------------------------------|
| `/api/meta`      | GET    | dropdown options, feature stats, metrics |
| `/api/findings`  | GET    | all visualisation data (real predictions)|
| `/api/history`   | GET    | a real `?area=&item=` window for autofill|
| `/api/predict`   | POST   | run the BiLSTM on a supplied window       |

---

## 8. Project layout

```
new_model3.0/
├── bilstm_yield_model.py   # end-to-end pipeline (data → model → eval)
├── app.py                  # Flask dashboard backend (loads model + scores)
├── templates/
│   └── index.html          # glass-morphism single-page UI
├── static/
│   ├── style.css           # glassmorphism theme
│   └── app.js              # charts + interactive tester
├── requirements.txt
├── README.md
└── outputs/                # generated on run
    ├── bilstm_yield_model.keras
    ├── preprocessing.pkl
    ├── metrics.json
    ├── training_history.png
    └── test_predictions.png
```

---

## 9. Design notes & honesty

- **No data leakage:** chronological split + train-only scaler fitting. The test
  years (2011–2013) are never seen during training or scaling.
- **Reproducible:** seeds fixed for NumPy and TensorFlow.
- **Contiguity enforced:** windows spanning missing years are discarded, so the
  BiLSTM only ever sees real consecutive seasons.
- Results above are the **actual output** of running `bilstm_yield_model.py`,
  not illustrative numbers.
