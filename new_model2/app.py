"""
================================================================================
 new_model2 Crop-Yield — Web Dashboard backend  (multi-model)
================================================================================
 A Flask app for the now-casting model suite. It loads ALL eight trained models,
 reproduces the exact inference each was trained with, and lets the dashboard
 switch between them — every chart, metric and live prediction follows the
 currently selected model. Default model: Option B (BiLSTM Neural Net).

 Run:  python app.py   ->   http://127.0.0.1:5001
================================================================================
"""

import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from sklearn.metrics import mean_squared_error, r2_score

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from tensorflow import keras

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PP_DIR    = os.path.join(BASE_DIR, "preprocessed_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


class SafeLabelEncoder:
    """Mirror of the preprocessing encoder so the pickle resolves under __main__."""
    def __init__(self, unseen_value="<UNKNOWN>"):
        self.unseen_value = unseen_value
        self.classes_ = None
        self.mapping = {}

    def transform(self, y):
        unk = self.mapping[self.unseen_value]
        return np.array([self.mapping.get(v, unk) for v in y])


app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# --------------------------------------------------------------------------- #
#  Artifacts
# --------------------------------------------------------------------------- #
print("Loading scalers / encoders / data ...")
with open(os.path.join(PP_DIR, "scalers.pkl"), "rb") as f:
    SC = pickle.load(f)
with open(os.path.join(PP_DIR, "encoders.pkl"), "rb") as f:
    ENC = pickle.load(f)
FEAT_SCALER, TARGET_SCALER, NUMF = SC["feature_scaler"], SC["target_scaler"], SC["numeric_features"]

DF = pd.read_csv(os.path.join(PP_DIR, "preprocessed_data.csv"))
DF = DF.sort_values(["Area", "Item", "Year"]).reset_index(drop=True)

EDITABLE = {
    "average_rain_fall_mm_per_year": "Rainfall (mm/yr)",
    "pesticides_tonnes": "Pesticides (t)",
    "avg_temp": "Avg temp (°C)",
    "hg/ha_yield_lag_1": "Yield 1 yr ago",
    "hg/ha_yield_lag_2": "Yield 2 yr ago",
    "hg/ha_yield_lag_3": "Yield 3 yr ago",
    "hg/ha_yield_lag_4": "Yield 4 yr ago",
    "hg/ha_yield_lag_5": "Yield 5 yr ago",
}
COUNTRIES = sorted(DF["Area"].unique().tolist())
CROPS     = sorted(DF["Item"].unique().tolist())

print("Loading 8 models ...")
def _load_pickle(name):
    with open(os.path.join(MODEL_DIR, name), "rb") as f:
        return pickle.load(f)

RF      = _load_pickle("rf_option_a.pkl")
XGB     = _load_pickle("xgb_option_a.pkl")
META    = _load_pickle("stacking_meta_option_a.pkl")
XGB_HB  = _load_pickle("xgb_hybrid_bilstm.pkl")
XGB_HU  = _load_pickle("xgb_hybrid_unilstm.pkl")
MLP     = keras.models.load_model(os.path.join(MODEL_DIR, "mlp_option_a.keras"))
BILSTM  = keras.models.load_model(os.path.join(MODEL_DIR, "bilstm_final.keras"))
UNILSTM = keras.models.load_model(os.path.join(MODEL_DIR, "unilstm_final.keras"))

# deep-feature extractors for the LSTM->XGBoost hybrids
def _extractor(model):
    return keras.Model(inputs=model.inputs,
                       outputs=[model.get_layer("lstm_out").output,
                                model.get_layer("country_flat").output,
                                model.get_layer("crop_flat").output])
EXT_BILSTM  = _extractor(BILSTM)
EXT_UNILSTM = _extractor(UNILSTM)

# display order = by quality (best first); default selection handled on frontend
MODELS = {
    "unilstm_nn":  "Option C: UniLSTM Neural Net",
    "bilstm_nn":   "Option B: BiLSTM Neural Net",
    "stacking":    "Option A: Stacking Hybrid",
    "unilstm_xgb": "Option C: UniLSTM-XGBoost Hybrid",
    "rf":          "Option A: Random Forest",
    "bilstm_xgb":  "Option B: BiLSTM-XGBoost Hybrid",
    "xgb":         "Option A: XGBoost",
    "mlp":         "Option A: MLP Embeddings",
}
DEFAULT_MODEL = "bilstm_nn"


# --------------------------------------------------------------------------- #
#  Inference — reproduce the exact training-time input layout
# --------------------------------------------------------------------------- #
def base_inputs(df_sub):
    x_num = FEAT_SCALER.transform(df_sub[NUMF].values.astype("float32")).astype("float32")
    country = ENC["Area"].transform(df_sub["Area"].values).reshape(-1, 1).astype("int32")
    crop    = ENC["Item"].transform(df_sub["Item"].values).reshape(-1, 1).astype("int32")
    X_full  = np.hstack([x_num, country, crop]).astype("float32")
    seq = np.zeros((len(df_sub), 5, 4), dtype="float32")
    for t in range(5):                       # timestep 0 -> lag_5 ... timestep 4 -> lag_1
        b = 4 + (5 - t - 1) * 4
        seq[:, t, :] = x_num[:, b:b + 4]
    curr = x_num[:, :4]
    return {"x_num": x_num, "country": country, "crop": crop,
            "X_full": X_full, "seq": seq, "curr": curr}


def predict_scaled(key, inp):
    if key == "rf":
        return RF.predict(inp["X_full"])
    if key == "xgb":
        return XGB.predict(inp["X_full"])
    if key == "mlp":
        return MLP.predict([inp["x_num"], inp["country"], inp["crop"]], verbose=0).flatten()
    if key == "stacking":
        rf_p  = RF.predict(inp["X_full"])
        xgb_p = XGB.predict(inp["X_full"])
        mlp_p = MLP.predict([inp["x_num"], inp["country"], inp["crop"]], verbose=0).flatten()
        return META.predict(np.column_stack([rf_p, xgb_p, mlp_p]))
    if key in ("bilstm_nn", "unilstm_nn"):
        m = BILSTM if key == "bilstm_nn" else UNILSTM
        return m.predict([inp["seq"], inp["curr"], inp["country"], inp["crop"]], verbose=0).flatten()
    if key in ("bilstm_xgb", "unilstm_xgb"):
        ext = EXT_BILSTM if key == "bilstm_xgb" else EXT_UNILSTM
        hb  = XGB_HB if key == "bilstm_xgb" else XGB_HU
        feats = ext.predict([inp["seq"], inp["curr"], inp["country"], inp["crop"]], verbose=0)
        return hb.predict(np.column_stack(list(feats) + [inp["curr"]]))
    raise KeyError(key)


def predict_raw(key, df_sub, inp=None):
    inp = inp or base_inputs(df_sub)
    p = predict_scaled(key, inp).reshape(-1, 1)
    return TARGET_SCALER.inverse_transform(p).flatten()


# --------------------------------------------------------------------------- #
#  Score every model once, over all rows
# --------------------------------------------------------------------------- #
print("Scoring all models over the dataset ...")
LABELS = DF[["Area", "Item", "Year", "hg/ha_yield"]].rename(columns={"hg/ha_yield": "actual"})
LABELS["split"] = np.where(DF["Year"] <= 2007, "train",
                  np.where(DF["Year"] <= 2010, "val", "test"))
_INP_ALL = base_inputs(DF)
SCORED = {}        # key -> DataFrame(area,item,year,actual,pred,split,residual)
LEADER = []        # per-model test metrics
for key, name in MODELS.items():
    pred = predict_raw(key, DF, _INP_ALL)
    d = LABELS.copy()
    d["pred"] = pred
    d["residual"] = d["pred"] - d["actual"]
    SCORED[key] = d
    t = d[d["split"] == "test"]
    ape = np.abs((t["pred"].to_numpy() - t["actual"].to_numpy()) / t["actual"].to_numpy())
    LEADER.append({"key": key, "name": name,
                   "r2": float(r2_score(t["actual"], t["pred"])),
                   "rmse": float(np.sqrt(mean_squared_error(t["actual"], t["pred"]))),
                   "acc20": float((ape <= 0.20).mean() * 100)})
    print(f"   {name:36s}  R2={LEADER[-1]['r2']:.4f}")
LEADER.sort(key=lambda d: d["r2"], reverse=True)
N_TEST = int((LABELS["split"] == "test").sum())
print(f"Ready. {len(MODELS)} models scored on {N_TEST:,} test rows.")


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    feat_stats = {c: {"label": EDITABLE[c], "mean": float(DF[c].mean())} for c in EDITABLE}
    return jsonify({
        "countries": COUNTRIES, "crops": CROPS,
        "editable": EDITABLE, "feature_stats": feat_stats,
        "models": LEADER, "default_model": DEFAULT_MODEL,
        "n_models": len(MODELS), "n_test": N_TEST,
    })


def _findings_for(d):
    test = d[d["split"] == "test"]
    sample = test.sample(min(900, len(test)), random_state=42)
    scatter = [{"x": float(a), "y": float(p)} for a, p in zip(sample["actual"], sample["pred"])]
    diag_max = float(max(test["actual"].max(), test["pred"].max()))

    per_crop = []
    for crop, g in test.groupby("Item"):
        if len(g) < 3:
            continue
        per_crop.append({"crop": crop, "r2": float(r2_score(g["actual"], g["pred"])),
                         "avg_yield": float(g["actual"].mean()), "n": int(len(g))})
    per_crop.sort(key=lambda x: x["r2"], reverse=True)

    counts = test["Area"].value_counts().head(12).index.tolist()
    per_country = []
    for c in counts:
        g = test[test["Area"] == c]
        per_country.append({"country": c,
                            "r2": float(r2_score(g["actual"], g["pred"])) if len(g) > 2 else None,
                            "avg_yield": float(g["actual"].mean()), "n": int(len(g))})
    per_country.sort(key=lambda x: x["avg_yield"], reverse=True)

    cnts, edges = np.histogram(test["residual"].to_numpy(), bins=30)
    residuals = {"counts": cnts.tolist(),
                 "centers": [float((edges[i] + edges[i + 1]) / 2) for i in range(len(edges) - 1)]}

    yearly = d.groupby("Year")[["actual", "pred"]].mean().reset_index().sort_values("Year")
    trend = {"years": [int(y) for y in yearly["Year"]],
             "actual": [float(v) for v in yearly["actual"]],
             "pred": [float(v) for v in yearly["pred"]]}

    ape = np.abs((test["pred"].to_numpy() - test["actual"].to_numpy()) / test["actual"].to_numpy())
    accuracy = {"within_10": float((ape <= 0.10).mean() * 100),
                "within_20": float((ape <= 0.20).mean() * 100),
                "within_30": float((ape <= 0.30).mean() * 100)}
    return {"scatter": scatter, "diag_max": diag_max, "per_crop": per_crop,
            "per_country": per_country, "residuals": residuals, "trend": trend,
            "accuracy": accuracy}


@app.route("/api/findings")
def api_findings():
    key = request.args.get("model", DEFAULT_MODEL)
    if key not in SCORED:
        return jsonify({"error": "unknown model"}), 400
    out = _findings_for(SCORED[key])
    info = next(m for m in LEADER if m["key"] == key)
    out["model"] = info
    return jsonify(out)


@app.route("/api/history")
def api_history():
    area, item = request.args.get("area"), request.args.get("item")
    g = DF[(DF["Area"] == area) & (DF["Item"] == item)].sort_values("Year")
    if g.empty:
        return jsonify({"available": False, "reason": "No data for this pair."})
    row = g.iloc[-1]
    return jsonify({"available": True, "year": int(row["Year"]),
                    "years": sorted({int(y) for y in g["Year"]}),
                    "values": {c: float(row[c]) for c in EDITABLE},
                    "actual": float(row["hg/ha_yield"])})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    area, item, year = data["area"], data["item"], int(data["year"])
    key = data.get("model", DEFAULT_MODEL)
    overrides = data.get("overrides", {})
    if key not in MODELS:
        return jsonify({"error": "unknown model"}), 400

    base = DF[(DF["Area"] == area) & (DF["Item"] == item) & (DF["Year"] == year)]
    if base.empty:
        return jsonify({"error": "No base row for this country/crop/year."}), 400
    row = base.iloc[[0]].copy()
    for col, val in overrides.items():
        if col in EDITABLE and val is not None:
            row[col] = float(val)
    try:
        pred = float(predict_raw(key, row)[0])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"bad input: {e}"}), 400
    return jsonify({"prediction": pred, "actual": float(base.iloc[0]["hg/ha_yield"]),
                    "year": year, "model": MODELS[key]})


if __name__ == "__main__":
    print("\n  Dashboard ready  ->  http://127.0.0.1:5001\n")
    app.run(host="127.0.0.1", port=5001, debug=False)
