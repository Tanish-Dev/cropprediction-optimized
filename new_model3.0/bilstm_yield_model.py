"""
================================================================================
 BiLSTM Crop-Yield Prediction Pipeline  (new_model3.0)
================================================================================
 A bidirectional-LSTM neural network that predicts crop yield (hg/ha) for a
 given country + crop, using a multi-year window of agro-climatic history.

 Pipeline
 --------
   1. Load + clean the FAO panel dataset (new_model2/yield_df.csv).
   2. Frame each (Country, Crop) pair as a temporal sequence and slice it into
      sliding windows  ->  X = last T years of features,  y = next year's yield.
   3. Leakage-safe scaling: scalers are fit ONLY on the training period.
   4. A hybrid Keras model:
         - a stacked Bidirectional-LSTM branch over the temporal window
         - learned Embeddings for the static Country / Crop identity
      are concatenated and passed through a dense regression head.
   5. Train with EarlyStopping + LR scheduling, then evaluate on a held-out
      future test period and export metrics + diagnostic plots.

 Author: Tanish  |  Framework: TensorFlow / Keras 3
================================================================================
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "new_model2", "yield_df.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW          = 4          # number of past years fed to the BiLSTM
TRAIN_END_YEAR  = 2007       # train  : target year <= 2007
VAL_END_YEAR    = 2010       # val    : 2008 - 2010   |  test : 2011+
SEQ_FEATURES    = ["hg/ha_yield", "average_rain_fall_mm_per_year",
                   "pesticides_tonnes", "avg_temp"]
TARGET          = "hg/ha_yield"
GROUP_KEYS      = ["Area", "Item"]

EPOCHS      = 120
BATCH_SIZE  = 256
SEED        = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------------------------------- #
#  1. Data loading & cleaning
# --------------------------------------------------------------------------- #
def load_data():
    print("[1/6] Loading dataset ...")
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)])

    # keep only physically valid records
    df = df[df["hg/ha_yield"] > 0].copy()
    df = df.drop_duplicates(subset=GROUP_KEYS + ["Year"])
    df = df.sort_values(GROUP_KEYS + ["Year"]).reset_index(drop=True)

    print(f"      rows: {len(df):,} | countries: {df['Area'].nunique()} | "
          f"crops: {df['Item'].nunique()} | years: {df['Year'].min()}-{df['Year'].max()}")
    return df


def encode_categoricals(df):
    """Integer-encode Country / Crop for embedding lookups."""
    encoders = {}
    for col in GROUP_KEYS:
        le = LabelEncoder()
        df[col + "_id"] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


# --------------------------------------------------------------------------- #
#  2. Sliding-window sequence construction
# --------------------------------------------------------------------------- #
def build_sequences(df):
    """
    For every (Country, Crop) series, build sliding windows of `WINDOW`
    consecutive years -> predict the yield of the following year.
    Only strictly contiguous year runs are used (no gaps).
    """
    print(f"[2/6] Building sliding-window sequences (window={WINDOW}) ...")
    X_seq, X_area, X_item, y, meta = [], [], [], [], []

    for (area, item), g in df.groupby(GROUP_KEYS):
        g = g.sort_values("Year")
        feats  = g[SEQ_FEATURES].values
        years  = g["Year"].values
        area_id = g["Area_id"].iloc[0]
        item_id = g["Item_id"].iloc[0]

        for i in range(len(g) - WINDOW):
            window_years = years[i:i + WINDOW + 1]
            # require a contiguous run of years (1990,1991,...)
            if window_years[-1] - window_years[0] != WINDOW:
                continue
            X_seq.append(feats[i:i + WINDOW])          # T past years
            y.append(feats[i + WINDOW, 0])             # next-year yield
            X_area.append(area_id)
            X_item.append(item_id)
            meta.append(years[i + WINDOW])             # target year (for split)

    X_seq = np.asarray(X_seq, dtype="float32")
    X_area = np.asarray(X_area, dtype="int32")
    X_item = np.asarray(X_item, dtype="int32")
    y = np.asarray(y, dtype="float32")
    meta = np.asarray(meta, dtype="int32")
    print(f"      built {len(y):,} sequences of shape {X_seq.shape[1:]} ")
    return X_seq, X_area, X_item, y, meta


# --------------------------------------------------------------------------- #
#  3. Temporal split + leakage-safe scaling
# --------------------------------------------------------------------------- #
def split_and_scale(X_seq, X_area, X_item, y, meta):
    print("[3/6] Temporal split + leakage-safe scaling ...")
    train = meta <= TRAIN_END_YEAR
    val   = (meta > TRAIN_END_YEAR) & (meta <= VAL_END_YEAR)
    test  = meta > VAL_END_YEAR

    # Fit feature scaler on TRAIN timesteps only, then transform every window.
    n_feat = X_seq.shape[2]
    feat_scaler = StandardScaler().fit(X_seq[train].reshape(-1, n_feat))

    def scale_seq(a):
        flat = feat_scaler.transform(a.reshape(-1, n_feat))
        return flat.reshape(a.shape).astype("float32")

    Xs = scale_seq(X_seq)

    target_scaler = StandardScaler().fit(y[train].reshape(-1, 1))
    ys = target_scaler.transform(y.reshape(-1, 1)).flatten().astype("float32")

    def pack(mask):
        return {
            "seq":  Xs[mask],
            "area": X_area[mask],
            "item": X_item[mask],
            "y":    ys[mask],
            "y_raw": y[mask],
            "year": meta[mask],
        }

    data = {"train": pack(train), "val": pack(val), "test": pack(test)}
    for k, d in data.items():
        print(f"      {k:5s}: {len(d['y']):,} sequences")
    return data, feat_scaler, target_scaler


# --------------------------------------------------------------------------- #
#  4. Model
# --------------------------------------------------------------------------- #
def build_model(n_area, n_item):
    print("[4/6] Building BiLSTM model ...")
    seq_in  = keras.Input(shape=(WINDOW, len(SEQ_FEATURES)), name="seq")
    area_in = keras.Input(shape=(1,), name="area", dtype="int32")
    item_in = keras.Input(shape=(1,), name="item", dtype="int32")

    # --- Temporal branch: stacked Bidirectional LSTM ----------------------- #
    x = layers.Bidirectional(
            layers.LSTM(96, return_sequences=True), name="bilstm_1")(seq_in)
    x = layers.Dropout(0.25)(x)
    x = layers.Bidirectional(
            layers.LSTM(48), name="bilstm_2")(x)
    x = layers.Dropout(0.25)(x)

    # --- Static identity branch: learned embeddings ------------------------ #
    area_emb = layers.Flatten()(
        layers.Embedding(n_area, min(32, (n_area + 1) // 2))(area_in))
    item_emb = layers.Flatten()(
        layers.Embedding(n_item, min(8, (n_item + 1) // 2))(item_in))

    # --- Fusion + regression head ------------------------------------------ #
    h = layers.Concatenate()([x, area_emb, item_emb])
    h = layers.Dense(128, activation="relu")(h)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.30)(h)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dropout(0.20)(h)
    out = layers.Dense(1, name="yield")(h)

    model = keras.Model([seq_in, area_in, item_in], out, name="BiLSTM_YieldNet")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    model.summary(print_fn=lambda s: print("      " + s))
    return model


# --------------------------------------------------------------------------- #
#  5. Train
# --------------------------------------------------------------------------- #
def train_model(model, data):
    print("[5/6] Training ...")
    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6,
            min_lr=1e-5, verbose=1),
    ]
    tr, va = data["train"], data["val"]
    hist = model.fit(
        {"seq": tr["seq"], "area": tr["area"], "item": tr["item"]}, tr["y"],
        validation_data=(
            {"seq": va["seq"], "area": va["area"], "item": va["item"]}, va["y"]),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cbs, verbose=2,
    )
    return hist


# --------------------------------------------------------------------------- #
#  6. Evaluate
# --------------------------------------------------------------------------- #
def evaluate(model, data, target_scaler, hist):
    print("[6/6] Evaluating ...")

    def predict(split):
        d = data[split]
        p = model.predict(
            {"seq": d["seq"], "area": d["area"], "item": d["item"]},
            verbose=0).flatten()
        return target_scaler.inverse_transform(p.reshape(-1, 1)).flatten()

    results = {}
    for split in ["train", "val", "test"]:
        y_true = data[split]["y_raw"]
        y_pred = predict(split)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))
        mask = y_true > 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        results[split] = {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_%": mape}
        print(f"      {split:5s} | R2={r2:6.4f} | RMSE={rmse:11.1f} | "
              f"MAE={mae:11.1f} | MAPE={mape:5.2f}%")

    # ---- persist metrics ---- #
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- diagnostic plots ---- #
    _plot_history(hist)
    _plot_predictions(data["test"]["y_raw"], predict("test"), results["test"])
    return results


def _plot_history(hist):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist.history["loss"], label="train")
    ax[0].plot(hist.history["val_loss"], label="val")
    ax[0].set_title("Loss (Huber)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist.history["mae"], label="train")
    ax[1].plot(hist.history["val_mae"], label="val")
    ax[1].set_title("MAE (scaled)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=130)
    plt.close(fig)


def _plot_predictions(y_true, y_pred, m):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, edgecolors="none")
    lim = [0, max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1.5, label="perfect")
    ax.set_xlabel("Actual yield (hg/ha)")
    ax.set_ylabel("Predicted yield (hg/ha)")
    ax.set_title(f"Test set  |  R² = {m['R2']:.3f}   MAPE = {m['MAPE_%']:.1f}%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "test_predictions.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
def main():
    print("=" * 80)
    print("  BiLSTM CROP-YIELD PREDICTION  (new_model3.0)")
    print("=" * 80)

    df = load_data()
    df, encoders = encode_categoricals(df)
    X_seq, X_area, X_item, y, meta = build_sequences(df)
    data, feat_scaler, target_scaler = split_and_scale(X_seq, X_area, X_item, y, meta)

    n_area = df["Area_id"].max() + 1
    n_item = df["Item_id"].max() + 1
    model = build_model(n_area, n_item)
    hist = train_model(model, data)
    results = evaluate(model, data, target_scaler, hist)

    # ---- persist artifacts ---- #
    model.save(os.path.join(OUTPUT_DIR, "bilstm_yield_model.keras"))
    with open(os.path.join(OUTPUT_DIR, "preprocessing.pkl"), "wb") as f:
        pickle.dump({"encoders": encoders,
                     "feature_scaler": feat_scaler,
                     "target_scaler": target_scaler,
                     "window": WINDOW,
                     "seq_features": SEQ_FEATURES}, f)

    print("=" * 80)
    print("  DONE.  Artifacts saved to:", OUTPUT_DIR)
    print(f"  Test R2 = {results['test']['R2']:.4f} | "
          f"Test MAPE = {results['test']['MAPE_%']:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
