"""
================================================================================
 BiLSTM Crop-Yield — Web Dashboard backend  (new_model3.0)
================================================================================
 A Flask app that:
   - loads the trained BiLSTM model + fitted scalers/encoders,
   - rebuilds every sliding-window sequence and scores it once at startup so the
     dashboard can render real findings (per-crop / per-country / residuals …),
   - exposes a JSON API consumed by a glass-morphism single-page frontend,
   - lets the user interactively test the model on any country + crop window.

 Run:  python app.py   ->   http://127.0.0.1:5000
================================================================================
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from sklearn.metrics import mean_squared_error, r2_score

# Reuse the EXACT pipeline logic the model was trained with.  Importing works
# because app.py is run from inside this folder (its dir is on sys.path).
from bilstm_yield_model import (
    load_data, encode_categoricals, build_sequences,
    SEQ_FEATURES, WINDOW, TRAIN_END_YEAR, VAL_END_YEAR, OUTPUT_DIR,
)

from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# --------------------------------------------------------------------------- #
#  Load artifacts + precompute findings (once, at import time)
# --------------------------------------------------------------------------- #
print("Loading model + artifacts ...")
MODEL = keras.models.load_model(os.path.join(OUTPUT_DIR, "bilstm_yield_model.keras"))
with open(os.path.join(OUTPUT_DIR, "preprocessing.pkl"), "rb") as f:
    PREP = pickle.load(f)
FEAT_SCALER   = PREP["feature_scaler"]
TARGET_SCALER = PREP["target_scaler"]
ENCODERS      = PREP["encoders"]
with open(os.path.join(OUTPUT_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

# Raw dataframe (for the interactive tester / history autofill)
DF = load_data()
DF, _ = encode_categoricals(DF)

CROPS     = sorted(DF["Item"].unique().tolist())
COUNTRIES = sorted(DF["Area"].unique().tolist())


def _scale_seq(arr):
    n = arr.shape[-1]
    flat = FEAT_SCALER.transform(arr.reshape(-1, n))
    return flat.reshape(arr.shape).astype("float32")


def _split_of(year):
    if year <= TRAIN_END_YEAR:
        return "train"
    if year <= VAL_END_YEAR:
        return "val"
    return "test"


print("Scoring every sequence for the findings views ...")
_Xseq, _Xarea, _Xitem, _y, _meta = build_sequences(DF)
_pred_scaled = MODEL.predict(
    {"seq": _scale_seq(_Xseq), "area": _Xarea, "item": _Xitem}, verbose=0).flatten()
_pred = TARGET_SCALER.inverse_transform(_pred_scaled.reshape(-1, 1)).flatten()

SCORED = pd.DataFrame({
    "area":   ENCODERS["Area"].inverse_transform(_Xarea),
    "item":   ENCODERS["Item"].inverse_transform(_Xitem),
    "year":   _meta,
    "actual": _y,
    "pred":   _pred,
})
SCORED["split"]    = SCORED["year"].apply(_split_of)
SCORED["residual"] = SCORED["pred"] - SCORED["actual"]
print(f"Ready. {len(SCORED):,} scored sequences.")


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    """Static metadata for the UI controls + headline numbers."""
    feat_stats = {
        f: {"min": float(DF[f].min()),
            "max": float(DF[f].max()),
            "mean": float(DF[f].mean())}
        for f in SEQ_FEATURES
    }
    return jsonify({
        "countries": COUNTRIES,
        "crops": CROPS,
        "window": WINDOW,
        "seq_features": SEQ_FEATURES,
        "feature_stats": feat_stats,
        "metrics": METRICS,
        "n_sequences": int(len(SCORED)),
        "year_range": [int(DF["Year"].min()), int(DF["Year"].max())],
        "split_years": {"train": f"<= {TRAIN_END_YEAR}",
                        "val": f"{TRAIN_END_YEAR + 1}-{VAL_END_YEAR}",
                        "test": f"{VAL_END_YEAR + 1}+"},
    })


@app.route("/api/findings")
def api_findings():
    """Everything the visualisation panels need, computed from real predictions."""
    test = SCORED[SCORED["split"] == "test"]

    # 1. Predicted-vs-actual scatter (sample for a responsive chart)
    sample = test.sample(min(900, len(test)), random_state=42)
    scatter = [{"x": float(a), "y": float(p)}
               for a, p in zip(sample["actual"], sample["pred"])]
    diag_max = float(max(test["actual"].max(), test["pred"].max()))

    # 2. Per-crop performance on the test set
    per_crop = []
    for crop, g in test.groupby("item"):
        if len(g) < 3:
            continue
        per_crop.append({
            "crop": crop,
            "r2": float(r2_score(g["actual"], g["pred"])),
            "rmse": float(np.sqrt(mean_squared_error(g["actual"], g["pred"]))),
            "avg_yield": float(g["actual"].mean()),
            "n": int(len(g)),
        })
    per_crop.sort(key=lambda d: d["r2"], reverse=True)

    # 3. Top-12 countries by sequence count (test set)
    counts = test["area"].value_counts().head(12).index.tolist()
    per_country = []
    for c in counts:
        g = test[test["area"] == c]
        per_country.append({
            "country": c,
            "r2": float(r2_score(g["actual"], g["pred"])) if len(g) > 2 else None,
            "avg_yield": float(g["actual"].mean()),
            "n": int(len(g)),
        })
    per_country.sort(key=lambda d: d["avg_yield"], reverse=True)

    # 4. Residual distribution (histogram)
    res = test["residual"].values
    counts_h, edges = np.histogram(res, bins=30)
    residuals = {
        "counts": counts_h.tolist(),
        "centers": [float((edges[i] + edges[i + 1]) / 2) for i in range(len(edges) - 1)],
    }

    # 5. Mean actual vs predicted yield per year (all splits) — trend view
    yearly = (SCORED.groupby("year")[["actual", "pred"]].mean()
              .reset_index().sort_values("year"))
    trend = {
        "years": [int(y) for y in yearly["year"]],
        "actual": [float(v) for v in yearly["actual"]],
        "pred": [float(v) for v in yearly["pred"]],
    }

    # 6. Accuracy: share of test predictions within tolerance bands
    ape = np.abs((test["pred"] - test["actual"]) / test["actual"])
    accuracy = {
        "within_10": float((ape <= 0.10).mean() * 100),
        "within_20": float((ape <= 0.20).mean() * 100),
        "within_30": float((ape <= 0.30).mean() * 100),
        "mean": float(100 - ape.mean() * 100),      # 100 - MAPE
        "median_ape": float(ape.median() * 100),
    }

    return jsonify({
        "scatter": scatter,
        "diag_max": diag_max,
        "per_crop": per_crop,
        "per_country": per_country,
        "residuals": residuals,
        "trend": trend,
        "accuracy": accuracy,
    })


@app.route("/api/history")
def api_history():
    """Return the WINDOW years preceding a target year for autofill + the actual."""
    area = request.args.get("area")
    item = request.args.get("item")
    g = DF[(DF["Area"] == area) & (DF["Item"] == item)].sort_values("Year")
    if g.empty:
        return jsonify({"available": False, "reason": "No data for this pair."})

    years = g["Year"].tolist()
    # pick the latest target year that has a full contiguous preceding window
    target = None
    for ty in reversed(years):
        win = [ty - k for k in range(WINDOW, 0, -1)]
        if all(y in years for y in win):
            target = ty
            break
    if target is None:
        return jsonify({"available": False,
                        "reason": "No contiguous 4-year window available."})

    rows = []
    for y in [target - k for k in range(WINDOW, 0, -1)]:
        r = g[g["Year"] == y].iloc[0]
        rows.append({"year": int(y), **{f: float(r[f]) for f in SEQ_FEATURES}})
    actual_row = g[g["Year"] == target].iloc[0]
    return jsonify({
        "available": True,
        "target_year": int(target),
        "window": rows,
        "actual": float(actual_row["hg/ha_yield"]),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Run the BiLSTM on a user-supplied window."""
    data = request.get_json(force=True)
    area = data["area"]
    item = data["item"]
    window = data["window"]      # list of WINDOW dicts with SEQ_FEATURES keys

    if len(window) != WINDOW:
        return jsonify({"error": f"window must have {WINDOW} rows"}), 400

    try:
        seq = np.array([[row[f] for f in SEQ_FEATURES] for row in window],
                       dtype="float32").reshape(1, WINDOW, len(SEQ_FEATURES))
        area_id = ENCODERS["Area"].transform([area]).astype("int32")
        item_id = ENCODERS["Item"].transform([item]).astype("int32")
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"bad input: {e}"}), 400

    pred_scaled = MODEL.predict(
        {"seq": _scale_seq(seq), "area": area_id, "item": item_id}, verbose=0)
    pred = float(TARGET_SCALER.inverse_transform(pred_scaled)[0, 0])
    return jsonify({"prediction": pred})


if __name__ == "__main__":
    print("\n  Dashboard ready  ->  http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
