"""
ICRISAT Crop Yield Prediction — Optimized Pipeline
====================================================
Best model: BiLSTM + Self-Attention + Spatial Embeddings
Baseline:   XGBoost + LightGBM with engineered lag features

Key improvements over standard BiLSTM:
  - Self-attention over LSTM outputs (captures which years matter most)
  - Huber loss (robust to yield outliers, which are common in ICRISAT)
  - Rolling statistics (3-yr mean/std) as extra temporal context
  - BatchNormalization + ReduceLROnPlateau for stable training
  - Proper tolerance-based accuracy (5%, 10%, 15%, 20%)
  - Unknown category handling for unseen districts/crops in val/test

Usage:
  python crop_yield_pipeline.py

Requires:
  pip install tensorflow xgboost lightgbm scikit-learn pandas numpy
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks as keras_callbacks
from tensorflow.keras.regularizers import l2

np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH   = "ICRISAT-District Level Data.csv"   # ← update path if needed
OUTPUT_DIR  = "pipeline_outputs"
N_LAGS      = 5
TRAIN_END   = 2010
VAL_START   = 2011
VAL_END     = 2013
TEST_START  = 2014

os.makedirs(OUTPUT_DIR, exist_ok=True)

BANNER = "=" * 65


# ─────────────────────────────────────────────────────────────────────
# 1. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────

def load_and_melt(path: str) -> pd.DataFrame:
    """Wide ICRISAT CSV → long format with one row per (State, District, Crop, Year)."""
    df = pd.read_csv(path)
    yield_cols = [c for c in df.columns if "YIELD" in c]
    crops = [c.replace(" YIELD (Kg per ha)", "") for c in yield_cols]

    records = []
    for crop in crops:
        area_col  = f"{crop} AREA (1000 ha)"
        yield_col = f"{crop} YIELD (Kg per ha)"
        if area_col not in df.columns:
            continue
        tmp = df[["State Name", "Dist Name", "Year", area_col, yield_col]].copy()
        tmp.columns = ["State", "District", "Year", "Area", "Yield"]
        tmp["Crop"] = crop
        records.append(tmp)

    long_df = pd.concat(records, ignore_index=True)

    # Drop inactive rows (-1 placeholder, zero area/yield)
    long_df = long_df[(long_df["Area"] > 0) & (long_df["Yield"] > 0)]
    long_df = long_df.reset_index(drop=True)
    return long_df


def engineer_features(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    """Add lag features + rolling statistics. Drops rows with incomplete lag history."""
    df = df.sort_values(["State", "District", "Crop", "Year"]).reset_index(drop=True)
    grp = df.groupby(["State", "District", "Crop"])

    for lag in range(1, n_lags + 1):
        df[f"yield_lag_{lag}"] = grp["Yield"].shift(lag)
        df[f"area_lag_{lag}"]  = grp["Area"].shift(lag)

    # 3-year rolling mean and std of yield (leakage-free: shift(1) before rolling)
    df["yield_roll_mean3"] = grp["Yield"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    df["yield_roll_std3"] = grp["Yield"].transform(
        lambda x: x.shift(1).rolling(3).std()
    )

    # Year-over-year delta (yield trend signal)
    df["yield_yoy_delta"] = grp["Yield"].transform(
        lambda x: x.shift(1) - x.shift(2)
    )

    df = df.dropna(subset=[f"yield_lag_{n_lags}"]).reset_index(drop=True)
    return df


def safe_label_encode(le: LabelEncoder, values):
    """Transform labels; map unseen values to a dedicated 'unknown' index."""
    mapping = {v: i for i, v in enumerate(le.classes_)}
    unknown_idx = len(le.classes_)   # one past the last known class
    return np.array([mapping.get(v, unknown_idx) for v in values])


def encode_and_split(df: pd.DataFrame):
    """Temporal split → leakage-free label encoding and scaling."""
    train = df[df["Year"] <= TRAIN_END].copy()
    val   = df[(df["Year"] >= VAL_START) & (df["Year"] <= VAL_END)].copy()
    test  = df[df["Year"] >= TEST_START].copy()

    # Label encode categoricals — fit only on train
    encoders = {}
    for col in ["State", "District", "Crop"]:
        le = LabelEncoder()
        train[col + "_enc"] = le.fit_transform(train[col])
        val[col + "_enc"]   = safe_label_encode(le, val[col])
        test[col + "_enc"]  = safe_label_encode(le, test[col])
        encoders[col] = le

    # Continuous feature columns
    lag_yield_cols  = [f"yield_lag_{i}" for i in range(1, N_LAGS + 1)]
    lag_area_cols   = [f"area_lag_{i}"  for i in range(1, N_LAGS + 1)]
    extra_feat_cols = ["yield_roll_mean3", "yield_roll_std3", "yield_yoy_delta", "Area", "Year"]
    all_feat_cols   = lag_yield_cols + lag_area_cols + extra_feat_cols

    # Scale — fit only on train
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    for split in [train, val, test]:
        split[all_feat_cols] = scaler_X.fit_transform(split[all_feat_cols]) \
            if split is train else scaler_X.transform(split[all_feat_cols])
        split["Yield_scaled"] = scaler_y.fit_transform(split[["Yield"]]) \
            if split is train else scaler_y.transform(split[["Yield"]])

    return train, val, test, encoders, scaler_X, scaler_y, lag_yield_cols, lag_area_cols, extra_feat_cols


# ─────────────────────────────────────────────────────────────────────
# 2. BILSTM + SELF-ATTENTION MODEL
# ─────────────────────────────────────────────────────────────────────

def build_bilstm_attention(n_states, n_districts, n_crops, n_extra_feats):
    """
    Architecture:
      Sequence input (5 timesteps × 2 features) → BiLSTM → MultiHeadAttention
      Spatial inputs (State, District, Crop)     → Embedding layers
      Extra tabular features                     → Direct Dense
      All concatenated → Dense head → Yield prediction
    """
    # ── Inputs ──
    seq_input      = layers.Input(shape=(N_LAGS, 2),   name="seq_input")
    state_input    = layers.Input(shape=(1,),           name="state_input")
    district_input = layers.Input(shape=(1,),           name="district_input")
    crop_input     = layers.Input(shape=(1,),           name="crop_input")
    extra_input    = layers.Input(shape=(n_extra_feats,), name="extra_input")

    # ── Temporal branch: BiLSTM + Self-Attention ──
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, recurrent_dropout=0.1)
    )(seq_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True)
    )(x)

    # Self-attention lets the model weight "which year in the 5-yr window matters most"
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16, dropout=0.1)(x, x)
    attn = layers.LayerNormalization()(attn + x)          # residual connection
    temporal_feat = layers.GlobalAveragePooling1D()(attn)

    # ── Spatial embeddings ──
    state_emb    = layers.Flatten()(layers.Embedding(n_states + 2,    8)(state_input))
    district_emb = layers.Flatten()(layers.Embedding(n_districts + 2, 16)(district_input))
    crop_emb     = layers.Flatten()(layers.Embedding(n_crops + 2,     8)(crop_input))

    # ── Concatenate all branches ──
    combined = layers.concatenate([temporal_feat, state_emb, district_emb, crop_emb, extra_input])

    # ── Dense prediction head ──
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1, name="yield_output")(x)

    model = Model(
        inputs=[seq_input, state_input, district_input, crop_input, extra_input],
        outputs=output
    )

    # Huber loss is more robust than MSE to extreme yield outliers
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=1.5),
        metrics=["mae"]
    )
    return model


def to_nn_inputs(df, lag_yield_cols, lag_area_cols, extra_feat_cols):
    """Package a dataframe into the 5-input dict the model expects."""
    yield_seq = df[lag_yield_cols].values          # (N, 5)
    area_seq  = df[lag_area_cols].values           # (N, 5)
    seq       = np.stack([yield_seq, area_seq], axis=-1)   # (N, 5, 2)

    return {
        "seq_input":      seq,
        "state_input":    df["State_enc"].values.reshape(-1, 1),
        "district_input": df["District_enc"].values.reshape(-1, 1),
        "crop_input":     df["Crop_enc"].values.reshape(-1, 1),
        "extra_input":    df[extra_feat_cols].values,
    }


# ─────────────────────────────────────────────────────────────────────
# 3. EVALUATION
# ─────────────────────────────────────────────────────────────────────

def tolerance_accuracy(y_true, y_pred, pct):
    """% of predictions within ±pct of the true value (relative error)."""
    mask = y_true > 0
    yt, yp = y_true[mask], y_pred[mask]
    return np.mean(np.abs(yt - yp) / yt <= pct) * 100


def evaluate(name, y_true, y_pred, results_list):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mbe   = float(np.mean(y_pred - y_true))
    r2    = r2_score(y_true, y_pred)
    acc5  = tolerance_accuracy(y_true, y_pred, 0.05)
    acc10 = tolerance_accuracy(y_true, y_pred, 0.10)
    acc15 = tolerance_accuracy(y_true, y_pred, 0.15)
    acc20 = tolerance_accuracy(y_true, y_pred, 0.20)

    print(f"\n  {'─'*58}")
    print(f"  {name}")
    print(f"  {'─'*58}")
    print(f"  RMSE      : {rmse:>10.2f} Kg/ha")
    print(f"  MAE       : {mae:>10.2f} Kg/ha")
    print(f"  MBE (Bias): {mbe:>10.2f} Kg/ha")
    print(f"  R²        : {r2:>10.4f}")
    print(f"  Acc_5%    : {acc5:>10.2f}%   ← within 5% of true yield")
    print(f"  Acc_10%   : {acc10:>10.2f}%   ← within 10% of true yield")
    print(f"  Acc_15%   : {acc15:>10.2f}%   ← within 15% of true yield  [PRIMARY]")
    print(f"  Acc_20%   : {acc20:>10.2f}%   ← within 20% of true yield")

    results_list.append({
        "Model": name, "RMSE": round(rmse, 2), "MAE": round(mae, 2),
        "MBE": round(mbe, 2), "R²": round(r2, 4),
        "Acc_5%": round(acc5, 2), "Acc_10%": round(acc10, 2),
        "Acc_15%": round(acc15, 2), "Acc_20%": round(acc20, 2),
    })


# ─────────────────────────────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────

def main():
    print(BANNER)
    print("  ICRISAT Crop Yield Prediction — Optimized Pipeline")
    print(BANNER)

    # ── Step 1: Preprocess ──────────────────────────────────────────
    print("\n[1/5] Loading and preprocessing data...")
    df = load_and_melt(DATA_PATH)
    df = engineer_features(df, n_lags=N_LAGS)
    (train, val, test,
     encoders, scaler_X, scaler_y,
     lag_yield_cols, lag_area_cols, extra_feat_cols) = encode_and_split(df)

    print(f"       Train : {len(train):>7,} samples  (1966 – {TRAIN_END})")
    print(f"       Val   : {len(val):>7,} samples  ({VAL_START} – {VAL_END})")
    print(f"       Test  : {len(test):>7,} samples  ({TEST_START} – 2017)")
    print(f"       Crops : {df['Crop'].nunique()}")

    n_states    = int(train["State_enc"].max())
    n_districts = int(train["District_enc"].max())
    n_crops     = int(train["Crop_enc"].max())

    # ── Step 2: Build & Train BiLSTM + Attention ─────────────────────
    print("\n[2/5] Training BiLSTM + Self-Attention model...")

    train_inputs = to_nn_inputs(train, lag_yield_cols, lag_area_cols, extra_feat_cols)
    val_inputs   = to_nn_inputs(val,   lag_yield_cols, lag_area_cols, extra_feat_cols)
    test_inputs  = to_nn_inputs(test,  lag_yield_cols, lag_area_cols, extra_feat_cols)

    model = build_bilstm_attention(n_states, n_districts, n_crops, len(extra_feat_cols))
    model.summary(line_length=70)

    cb = [
        keras_callbacks.EarlyStopping(
            patience=12, restore_best_weights=True, monitor="val_loss"
        ),
        keras_callbacks.ReduceLROnPlateau(
            factor=0.4, patience=5, min_lr=1e-5, monitor="val_loss"
        ),
        keras_callbacks.ModelCheckpoint(
            f"{OUTPUT_DIR}/bilstm_best.keras", save_best_only=True, monitor="val_loss"
        ),
    ]

    history = model.fit(
        train_inputs, train["Yield_scaled"].values,
        validation_data=(val_inputs, val["Yield_scaled"].values),
        epochs=120,
        batch_size=512,
        callbacks=cb,
        verbose=1,
    )

    model.save(f"{OUTPUT_DIR}/bilstm_attention_model.keras")

    # ── Step 3: XGBoost baseline ─────────────────────────────────────
    print("\n[3/5] Training XGBoost baseline...")
    xgb_feats = (lag_yield_cols + lag_area_cols + extra_feat_cols
                 + ["State_enc", "District_enc", "Crop_enc"])

    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
        eval_metric="rmse",
    )
    xgb_model.fit(
        train[xgb_feats], train["Yield_scaled"],
        eval_set=[(val[xgb_feats], val["Yield_scaled"])],
        verbose=200,
    )
    xgb_model.save_model(f"{OUTPUT_DIR}/xgboost_model.json")

    # ── Step 4: LightGBM baseline ────────────────────────────────────
    print("\n[4/5] Training LightGBM baseline...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        train[xgb_feats], train["Yield_scaled"],
        eval_set=[(val[xgb_feats], val["Yield_scaled"])],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(200)],
    )
    lgb_model.booster_.save_model(f"{OUTPUT_DIR}/lightgbm_model.txt")

    # ── Step 5: Evaluate on Test Set ─────────────────────────────────
    print("\n[5/5] Evaluation on Test Set (2014 – 2017)")
    print(BANNER)

    y_true = test["Yield"].values
    results = []

    # BiLSTM + Attention
    bilstm_scaled = model.predict(test_inputs, verbose=0).flatten()
    bilstm_pred   = scaler_y.inverse_transform(bilstm_scaled.reshape(-1, 1)).flatten()
    evaluate("BiLSTM + Self-Attention (Option B)", y_true, bilstm_pred, results)

    # XGBoost
    xgb_scaled = xgb_model.predict(test[xgb_feats])
    xgb_pred   = scaler_y.inverse_transform(xgb_scaled.reshape(-1, 1)).flatten()
    evaluate("XGBoost Baseline (Option A)", y_true, xgb_pred, results)

    # LightGBM
    lgb_scaled = lgb_model.predict(test[xgb_feats])
    lgb_pred   = scaler_y.inverse_transform(lgb_scaled.reshape(-1, 1)).flatten()
    evaluate("LightGBM Baseline (Option A)", y_true, lgb_pred, results)

    # Ensemble: average BiLSTM + LightGBM (often the sweet spot)
    ensemble_pred = 0.55 * bilstm_pred + 0.45 * lgb_pred
    evaluate("Ensemble (BiLSTM 55% + LightGBM 45%)", y_true, ensemble_pred, results)

    # ── Summary Table ────────────────────────────────────────────────
    print(f"\n{BANNER}")
    print("  SUMMARY TABLE")
    print(BANNER)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{OUTPUT_DIR}/evaluation_results.csv", index=False)

    best = results_df.loc[results_df["R²"].idxmax()]
    print(f"\n  Best model : {best['Model']}")
    print(f"  R²         : {best['R²']}")
    print(f"  RMSE       : {best['RMSE']} Kg/ha")
    print(f"  Acc_15%    : {best['Acc_15%']}%")
    print(f"  Acc_20%    : {best['Acc_20%']}%")
    print(f"\n  Results saved to → {OUTPUT_DIR}/evaluation_results.csv")
    print(BANNER)

    # Save artifacts
    with open(f"{OUTPUT_DIR}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(f"{OUTPUT_DIR}/scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open(f"{OUTPUT_DIR}/scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)


if __name__ == "__main__":
    main()
