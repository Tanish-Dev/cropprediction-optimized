"""
=============================================================================
Stage 2: Option A Model Training - Tabular Ensemble & Stacking Hybrid
=============================================================================
Trains three base tabular regressors:
  1. Random Forest Regressor (Baseline)
  2. XGBoost Regressor (Gradient Boosting)
  3. Multi-Layer Perceptron (MLP) Neural Network (with Embedding layers)
Combines predictions using a Stacking Meta-Regressor (Ridge) fit on Val set.
=============================================================================
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(BASE_DIR, 'preprocessed_data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_encoders_and_scalers():
    with open(os.path.join(PREPROCESSED_DIR, 'encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    with open(os.path.join(PREPROCESSED_DIR, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    return encoders, scalers


def build_mlp_model(num_states, num_districts, num_crops, num_features, lr=0.001):
    """
    Build MLP Neural Network with Embedding layers for categorical inputs.
    """
    # ── Inputs ──
    numeric_input = layers.Input(shape=(num_features,), name='mlp_numeric_input')
    state_input = layers.Input(shape=(1,), name='state_input')
    dist_input = layers.Input(shape=(1,), name='dist_input')
    crop_input = layers.Input(shape=(1,), name='crop_input')
    
    # ── Embedding Layers ──
    state_emb = layers.Embedding(input_dim=num_states, output_dim=8, name='state_emb')(state_input)
    dist_emb = layers.Embedding(input_dim=num_districts, output_dim=16, name='dist_emb')(dist_input)
    crop_emb = layers.Embedding(input_dim=num_crops, output_dim=8, name='crop_emb')(crop_input)
    
    # Flatten embeddings
    state_flat = layers.Flatten()(state_emb)
    dist_flat = layers.Flatten()(dist_emb)
    crop_flat = layers.Flatten()(crop_emb)
    
    # ── Concatenate ──
    x = layers.Concatenate()([numeric_input, state_flat, dist_flat, crop_flat])
    
    # ── Dense Regression Layers ──
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(32, activation='relu')(x)
    
    output = layers.Dense(1, name='yield_output')(x)
    
    model = Model(inputs=[numeric_input, state_input, dist_input, crop_input], outputs=output, name='MLP_Embed')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    return model


def run_training_option_a(epochs=100, batch_size=256, n_estimators=100, learning_rate=0.001):
    print("="*80)
    print("  CROP YIELD PREDICTION — OPTION A TRAINING")
    print("  Stage 2: Train RF, XGBoost, MLP & Blending Stacking Ensemble")
    print("="*80)

    # 1. Load preprocessed splits
    print("\n[1/5] Loading preprocessed split datasets...")
    splits = np.load(os.path.join(PREPROCESSED_DIR, 'train_test_splits.npz'))
    X_train = splits['X_train']
    y_train = splits['y_train']
    X_val = splits['X_val']
    y_val = splits['y_val']
    X_test = splits['X_test']
    y_test = splits['y_test']

    encoders, scalers = load_encoders_and_scalers()
    num_states = len(encoders['State Name'].classes_)
    num_districts = len(encoders['Dist Name'].classes_)
    num_crops = len(encoders['Crop'].classes_)
    
    # Features split into numeric (indices 0..11) and categorical (indices 12..14)
    # Train features
    X_train_num = X_train[:, :12]
    X_train_state = X_train[:, 12:13]
    X_train_dist = X_train[:, 13:14]
    X_train_crop = X_train[:, 14:15]
    
    # Val features
    X_val_num = X_val[:, :12]
    X_val_state = X_val[:, 12:13]
    X_val_dist = X_val[:, 13:14]
    X_val_crop = X_val[:, 14:15]

    # Test features
    X_test_num = X_test[:, :12]
    X_test_state = X_test[:, 12:13]
    X_test_dist = X_test[:, 13:14]
    X_test_crop = X_test[:, 14:15]

    print(f"    Train: {X_train.shape[0]:,} samples | Val: {X_val.shape[0]:,} samples | Test: {X_test.shape[0]:,} samples")

    # 2. Train Random Forest baseline
    print("\n[2/5] Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_split=8,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_val_preds = rf_model.predict(X_val)
    print("    ✓ RF trained successfully")

    # 3. Train XGBoost Regressor
    print("\n[3/5] Training XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_val_preds = xgb_model.predict(X_val)
    print("    ✓ XGBoost trained successfully")

    # 4. Train MLP Neural Network (Keras Embeddings)
    print("\n[4/5] Training MLP Neural Network with Categorical Embeddings...")
    mlp_model = build_mlp_model(num_states, num_districts, num_crops, num_features=12, lr=learning_rate)
    
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_DIR, 'mlp_option_a_best.keras'), 
                                  monitor='val_loss', save_best_only=True, verbose=0)
    ]
    
    mlp_model.fit(
        x=[X_train_num, X_train_state, X_train_dist, X_train_crop],
        y=y_train,
        validation_data=([X_val_num, X_val_state, X_val_dist, X_val_crop], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1
    )
    
    mlp_val_preds = mlp_model.predict([X_val_num, X_val_state, X_val_dist, X_val_crop]).flatten()
    print("    ✓ MLP Neural Network trained successfully")

    # 5. Build Stacking Ensemble using Ridge Regression on Validation Predictions
    print("\n[5/5] Building Stacking Ensemble...")
    # Stack validation predictions to train the meta-regressor
    val_pred_matrix = np.column_stack([rf_val_preds, xgb_val_preds, mlp_val_preds])
    
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(val_pred_matrix, y_val)
    
    print(f"    Meta-Regressor Coefficients (Blending Weights):")
    print(f"      RF Weight:  {meta_model.coef_[0]:.4f}")
    print(f"      XGB Weight: {meta_model.coef_[1]:.4f}")
    print(f"      MLP Weight: {meta_model.coef_[2]:.4f}")
    print(f"      Intercept:  {meta_model.intercept_:.4f}")

    # Generate test predictions for all base models & stacking ensemble
    rf_test_preds = rf_model.predict(X_test)
    xgb_test_preds = xgb_model.predict(X_test)
    mlp_test_preds = mlp_model.predict([X_test_num, X_test_state, X_test_dist, X_test_crop]).flatten()
    
    test_pred_matrix = np.column_stack([rf_test_preds, xgb_test_preds, mlp_test_preds])
    stacking_test_preds = meta_model.predict(test_pred_matrix)

    # Save Option A model artifacts
    with open(os.path.join(MODEL_DIR, 'rf_option_a.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)
    with open(os.path.join(MODEL_DIR, 'xgb_option_a.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)
    # MLP was already saved as keras model in callbacks, but let's make sure it is saved:
    mlp_model.save(os.path.join(MODEL_DIR, 'mlp_option_a.keras'))
    with open(os.path.join(MODEL_DIR, 'stacking_meta_option_a.pkl'), 'wb') as f:
        pickle.dump(meta_model, f)

    # Export intermediate predictions for final evaluation stage
    np.savez(os.path.join(MODEL_DIR, 'option_a_predictions.npz'),
             rf_preds=rf_test_preds,
             xgb_preds=xgb_test_preds,
             mlp_preds=mlp_test_preds,
             stacking_preds=stacking_test_preds)

    print("\n" + "="*80)
    print("  OPTION A TRAINING COMPLETE")
    print(f"  All models saved in: {MODEL_DIR}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    run_training_option_a(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_estimators=args.n_estimators,
        learning_rate=args.lr
    )
