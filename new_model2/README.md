# 🌾 Crop-Yield Intelligence — `new_model2`

An **8-model now-casting suite** that estimates crop yield (`hg/ha`) for a given
country + crop + year, using that year's agro-climatic conditions plus five years
of recent history. Includes classic tabular models, LSTM neural nets, and
LSTM→XGBoost hybrids — all compared head-to-head — and a **glass-morphism web
dashboard** to explore the findings and test the best model live.

> **Best model — Option C: Unidirectional-LSTM neural net (test set 2011–2013):**
> **R² = 0.985 · RMSE = 11,967 hg/ha · 87.9 % of predictions within ±20 %**

---

## 1. What this model does

For a `(country, crop, year)`, it estimates the yield from:

- the **current year's** rainfall, pesticide tonnage, temperature (and the year), and
- **5-year lag features** for yield, rainfall, pesticides and temperature.

Because it is given the **target year's own conditions**, this is **now-casting**
(explaining the present), not forecasting the future. Its sibling project
[`new_model3.0`](../new_model3.0) is a true one-year-ahead **forecast** — see the
comparison in §6.

---

## 2. Dataset

Source: the FAO panel (`yield_df.csv`, merged from `rainfall.csv`, `pesticides.csv`,
`temp.csv`, `yield.csv`).

| Property           | Value                                            |
|--------------------|--------------------------------------------------|
| Countries (`Area`) | 101                                              |
| Crops (`Item`)     | 10                                               |
| Years              | 1990 – 2013                                       |
| Target             | `hg/ha_yield`                                     |
| Features           | rainfall, pesticides, temperature + 5-year lags  |

**Leakage-safe temporal split:** train 1990–2007, validation 2008–2010,
**test 2011–2013** (3,439 samples). Scalers are fit on the training period only;
a `SafeLabelEncoder` maps unseen countries/crops to an `<UNKNOWN>` token.

---

## 3. The eight models

| # | Model | Strategy |
|---|-------|----------|
| A | Random Forest | tabular on flattened lag features |
| A | XGBoost | gradient-boosted trees |
| A | MLP Embeddings | dense net + country/crop embeddings |
| A | Stacking Hybrid | meta-learner over A models |
| B | **BiLSTM** Neural Net | bidirectional LSTM + embeddings |
| C | **UniLSTM** Neural Net | unidirectional LSTM + embeddings ⭐ **best** |
| B | BiLSTM → XGBoost Hybrid | deep LSTM features → XGBoost |
| C | UniLSTM → XGBoost Hybrid | deep LSTM features → XGBoost |

### Leaderboard (test set, by R²)

| Model | R² | RMSE | MAE | Within ±20 % |
|-------|----|------|-----|--------------|
| **Option C: UniLSTM Neural Net** | **0.9846** | **11,967** | 5,489 | **87.9 %** |
| Option B: BiLSTM Neural Net | 0.9842 | 12,106 | 5,586 | 82.0 % |
| Option A: Stacking Hybrid | 0.9836 | 12,328 | 5,481 | 86.6 % |
| Option C: UniLSTM-XGBoost Hybrid | 0.9799 | 13,662 | 6,490 | 88.5 % |
| Option A: Random Forest | 0.9798 | 13,698 | 5,543 | 90.6 % |
| Option B: BiLSTM-XGBoost Hybrid | 0.9789 | 13,973 | 6,354 | 88.4 % |
| Option A: XGBoost | 0.9776 | 14,409 | 5,938 | 90.7 % |
| Option A: MLP Embeddings | 0.9774 | 14,475 | 9,438 | 64.8 % |

---

## 4. The winning architecture (Option C)

```
   seq (5×4)        current (4)        country          crop
       │                │                 │               │
 ┌─────▼─────┐          │          ┌──────▼──────┐ ┌──────▼──────┐
 │ LSTM 32   │          │          │ Embed →16   │ │ Embed →8    │
 │ LSTM 16   │          │          └──────┬──────┘ └──────┬──────┘
 └─────┬─────┘          │                 │               │
   BatchNorm            └────────┬────────┴───────────────┘
       └──────────────── Concatenate
                              │
                  Dense 128 → Dropout → BN
                  Dense 64  → Dropout → BN   (L2 regularised)
                              │
                          Dense 1  →  yield
```

- **Sequence branch** — LSTM (32→16) over the 5-year window of `[yield, rainfall,
  pesticides, temp]` lags.
- **Embeddings** — country→16, crop→8 dimensions for location/crop baselines.
- **Current-year inputs** — year, rainfall, pesticides, temperature.
- Loss MSE · Adam · EarlyStopping + ReduceLROnPlateau.

---

## 5. Interactive web dashboard

A **light glass-morphism dashboard** (Flask + Chart.js + Lucide SVG icons — no
emojis) for exploring the findings and testing the best model.

```bash
source ../.venv/bin/activate
python app.py            # -> http://127.0.0.1:5001
```

> Requires the trained artifacts in `models/` and `preprocessed_data/`
> (produced by the pipeline scripts `01`–`04`). On startup the app loads **all 8
> models** and scores every row, so the charts show real predictions and exactly
> reproduce each model's reported metrics (verified to 4 decimals).

### Model picker — the dashboard's centrepiece

A **model picker** sits at the top of the findings. Choose any of the 8 models and
**every chart, KPI and live prediction updates to that model** — the active model's
name, R² and ±20 % accuracy are always shown in the picker bar, and the leaderboard
highlights it. The default model is **Option B: BiLSTM Neural Net**.

**Panels** (all follow the selected model):

| Panel | What it shows |
|-------|----------------|
| Model picker bar | the active model + its R² / ±20 % accuracy |
| KPI cards | selected model R² / accuracy / RMSE + models compared |
| **Model leaderboard** | Test R² across all 8 models, selected one highlighted |
| Predicted vs Actual | scatter of the selected model on the test set |
| Prediction accuracy | gauge + ±10/20/30 % tolerance bars |
| Accuracy by crop | per-crop R² |
| Residual distribution | error spread & symmetry |
| Yearly mean yield | actual vs predicted trend |
| Top-countries table | average yield + per-country fit |
| **Test the Model** | pick a country + crop → one-click estimate (with the **selected** model) vs the held-out actual; an "Adjust the inputs manually" panel lets you edit current-year conditions and recent yield history. Switch the active model to compare predictions on the same input. |

**API:** `GET /api/meta`, `GET /api/findings?model=<key>`, `GET /api/history`,
`POST /api/predict` (body includes `model`).

---

## 6. now-casting vs. forecasting — how it compares to `new_model3.0`

| Model | Task | Test R² | Within ±20 % |
|-------|------|---------|--------------|
| **`new_model2` — UniLSTM (this, best)** | **now-cast** | **0.985** | **87.9 %** |
| `new_model2` — BiLSTM | now-cast | 0.984 | 82.0 % |
| `new_model3.0` — BiLSTM | 1-yr forecast | 0.953 | 67.9 % |

`new_model2` is **given the target year's own** rainfall, pesticides and temperature,
so it solves a present-tense question and scores higher. `new_model3.0` sees **only
the past** and predicts a year it has never observed — a strictly harder task.
The higher numbers here reflect an **easier problem**, not universal superiority:

- use **now-casting** (this model) to *explain / fill-in* a year's yield given its weather;
- use **forecasting** (`new_model3.0`) to *plan ahead* before the year's weather is known.

---

## 7. Project layout

```
new_model2/
├── 01_preprocessing.py        # clean, lag-engineer, encode, scale, split
├── 02_train_option_a.py       # RF / XGBoost / MLP / Stacking
├── 03_train_lstm_hybrids.py   # BiLSTM & UniLSTM nets + XGBoost hybrids
├── 04_evaluation.py           # leaderboard, plots, report
├── main.py                    # runs the full pipeline
├── app.py                     # glass-morphism dashboard backend
├── templates/index.html       # dashboard UI
├── static/{style.css, app.js} # theme + charts + tester
├── README.md
├── preprocessed_data/         # splits, scalers, encoders
├── models/                    # trained models (best = unilstm_final.keras)
└── evaluation_output/         # metrics CSV, report, plots
```

> Note: an earlier `dashboard.py` (Dash-based) also exists; `app.py` is the new
> glass-morphism dashboard described above.
