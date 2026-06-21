"""
=============================================================================
Stage 3: Option B & C Model Training - LSTM Hybrids
=============================================================================
Trains two LSTM Keras architectures:
  - Option B: Bidirectional LSTM Branch + Categorical Embeddings
  - Option C: Unidirectional LSTM Branch + Categorical Embeddings
Extracts deep spatial-temporal features and trains XGBoost regressors on top
to form the final hybrid estimators.
=============================================================================
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
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


def prepare_lstm_sequences(X_num):
    """
    Construct sequence inputs from lagged numeric columns.
    Input shape: (samples, 12)
    Output shape: (samples, 5, 2)
    Timestep ordering: Chronological (5 years ago to 1 year ago)
    """
    samples = X_num.shape[0]
    seq = np.zeros((samples, 5, 2))
    
    # Timestep 0 (5 years ago): [yield_lag_5, area_lag_5]
    seq[:, 0, 0] = X_num[:, 6]  # yield_lag_5
    seq[:, 0, 1] = X_num[:, 11] # area_lag_5
    
    # Timestep 1 (4 years ago): [yield_lag_4, area_lag_4]
    seq[:, 1, 0] = X_num[:, 5]  # yield_lag_4
    seq[:, 1, 1] = X_num[:, 10] # area_lag_4
    
    # Timestep 2 (3 years ago): [yield_lag_3, area_lag_3]
    seq[:, 2, 0] = X_num[:, 4]  # yield_lag_3
    seq[:, 2, 1] = X_num[:, 9]  # area_lag_3
    
    # Timestep 3 (2 years ago): [yield_lag_2, area_lag_2]
    seq[:, 3, 0] = X_num[:, 3]  # yield_lag_2
    seq[:, 3, 1] = X_num[:, 8]  # area_lag_2
    
    # Timestep 4 (1 year ago): [yield_lag_1, area_lag_1]
    seq[:, 4, 0] = X_num[:, 2]  # yield_lag_1
    seq[:, 4, 1] = X_num[:, 7]  # area_lag_1
    
    return seq


def build_lstm_hybrid_model(num_states, num_districts, num_crops, bidirectional=True, lr=0.001):
    """
    Build LSTM-Embedding hybrid architecture.
    - Bidirectional LSTM (Option B) vs. Unidirectional LSTM (Option C).
    - Categorical variables mapped to Embeddings.
    """
    # ── Inputs ──
    seq_input = layers.Input(shape=(5, 2), name='lstm_sequence_input')
    numeric_input = layers.Input(shape=(2,), name='current_numeric_input') # Year, Area
    state_input = layers.Input(shape=(1,), name='state_input')
    dist_input = layers.Input(shape=(1,), name='dist_input')
    crop_input = layers.Input(shape=(1,), name='crop_input')
    
    # ── LSTM Branch ──
    if bidirectional:
        x_seq = layers.Bidirectional(layers.LSTM(32, return_sequences=True, name='lstm_1'))(seq_input)
        x_seq = layers.Dropout(0.2)(x_seq)
        x_seq = layers.Bidirectional(layers.LSTM(16, return_sequences=False, name='lstm_2'), name='lstm_out')(x_seq)
    else:
        x_seq = layers.LSTM(32, return_sequences=True, name='lstm_1')(seq_input)
        x_seq = layers.Dropout(0.2)(x_seq)
        x_seq = layers.LSTM(16, return_sequences=False, name='lstm_out')(x_seq)
        
    x_seq = layers.BatchNormalization()(x_seq)
    
    # ── Embeddings Branch ──
    state_emb = layers.Embedding(input_dim=num_states, output_dim=8, name='state_emb')(state_input)
    dist_emb = layers.Embedding(input_dim=num_districts, output_dim=16, name='dist_emb')(dist_input)
    crop_emb = layers.Embedding(input_dim=num_crops, output_dim=8, name='crop_emb')(crop_input)
    
    state_flat = layers.Flatten(name='state_flat')(state_emb)
    dist_flat = layers.Flatten(name='dist_flat')(dist_emb)
    crop_flat = layers.Flatten(name='crop_flat')(crop_emb)
    
    # ── Concatenation ──
    x = layers.Concatenate(name='concat_layer')([x_seq, state_flat, dist_flat, crop_flat, numeric_input])
    
    # ── Dense Prediction Head ──
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    
    output = layers.Dense(1, name='yield_output')(x)
    
    model_name = 'Option_B_BiLSTM' if bidirectional else 'Option_C_UniLSTM'
    model = Model(inputs=[seq_input, numeric_input, state_input, dist_input, crop_input], 
                  outputs=output, name=model_name)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    return model


def train_lstm_pipeline(bidirectional, epochs, batch_size, n_estimators, learning_rate,
                       X_train_seq, X_train_curr, X_train_state, X_train_dist, X_train_crop, y_train,
                       X_val_seq, X_val_curr, X_val_state, X_val_dist, X_val_crop, y_val,
                       X_test_seq, X_test_curr, X_test_state, X_test_dist, X_test_crop, y_test,
                       num_states, num_districts, num_crops):
    
    model_name = 'bilstm' if bidirectional else 'unilstm'
    print(f"\nTraining Neural Net for Option {'B (Bidirectional)' if bidirectional else 'C (Unidirectional)'}...")
    
    nn_model = build_lstm_hybrid_model(num_states, num_districts, num_crops, bidirectional, learning_rate)
    
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_DIR, f'{model_name}_best.keras'), 
                                  monitor='val_loss', save_best_only=True, verbose=0)
    ]
    
    nn_model.fit(
        x=[X_train_seq, X_train_curr, X_train_state, X_train_dist, X_train_crop],
        y=y_train,
        validation_data=([X_val_seq, X_val_curr, X_val_state, X_val_dist, X_val_crop], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1
    )
    
    nn_model.save(os.path.join(MODEL_DIR, f'{model_name}_final.keras'))
    nn_test_preds = nn_model.predict([X_test_seq, X_test_curr, X_test_state, X_test_dist, X_test_crop]).flatten()
    print(f"  ✓ {model_name} Neural Network trained.")

    # ── Feature Extraction for XGBoost Hybrid ──
    print(f"  Extracting deep features for XGBoost...")
    # Define an extractor model that outputs the outputs of LSTM & Flattened Embeddings
    extractor = Model(
        inputs=nn_model.inputs,
        outputs=[
            nn_model.get_layer('lstm_out').output,
            nn_model.get_layer('state_flat').output,
            nn_model.get_layer('dist_flat').output,
            nn_model.get_layer('crop_flat').output
        ]
    )
    
    def extract_features(seq, curr, state, dist, crop):
        feats = extractor.predict([seq, curr, state, dist, crop], verbose=0)
        # Concatenate intermediate outputs with current numeric features [Year, Area]
        return np.column_stack(list(feats) + [curr])

    X_train_extracted = extract_features(X_train_seq, X_train_curr, X_train_state, X_train_dist, X_train_crop)
    X_val_extracted   = extract_features(X_val_seq, X_val_curr, X_val_state, X_val_dist, X_val_crop)
    X_test_extracted  = extract_features(X_test_seq, X_test_curr, X_test_state, X_test_dist, X_test_crop)
    
    print(f"    Extracted features dimension: {X_train_extracted.shape[1]}")

    # ── Train XGBoost Regressor on Extracted Features ──
    print(f"  Training XGBoost on extracted features...")
    xgb_hybrid = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_hybrid.fit(
        X_train_extracted, y_train,
        eval_set=[(X_val_extracted, y_val)],
        verbose=False
    )
    
    xgb_test_preds = xgb_hybrid.predict(X_test_extracted)
    print("    ✓ XGBoost hybrid trained successfully")

    # Save XGBoost Hybrid
    with open(os.path.join(MODEL_DIR, f'xgb_hybrid_{model_name}.pkl'), 'wb') as f:
        pickle.dump(xgb_hybrid, f)
        
    return nn_test_preds, xgb_test_preds


def run_training_lstm_hybrids(epochs=100, batch_size=256, n_estimators=100, learning_rate=0.001):
    print("="*80)
    print("  CROP YIELD PREDICTION — OPTION B & C TRAINING")
    print("  Stage 3: Train Bidirectional LSTM (B) & Unidirectional LSTM (C) Hybrids")
    print("="*80)

    # Load splits
    print("\n[1/4] Loading splits...")
    splits = np.load(os.path.join(PREPROCESSED_DIR, 'train_test_splits.npz'))
    X_train = splits['X_train']
    y_train = splits['y_train']
    X_val   = splits['X_val']
    y_val   = splits['y_val']
    X_test  = splits['X_test']
    y_test  = splits['y_test']

    encoders, scalers = load_encoders_and_scalers()
    num_states = len(encoders['State Name'].classes_)
    num_districts = len(encoders['Dist Name'].classes_)
    num_crops = len(encoders['Crop'].classes_)

    # Prepare inputs
    print("\n[2/4] Restructuring numeric inputs into 5-year sequences...")
    # Numeric features (indices 0..11)
    # Categorical features (indices 12..14)
    
    # Train
    X_train_num = X_train[:, :12]
    X_train_seq = prepare_lstm_sequences(X_train_num)
    X_train_curr = X_train_num[:, :2] # Year, Area
    X_train_state = X_train[:, 12:13]
    X_train_dist = X_train[:, 13:14]
    X_train_crop = X_train[:, 14:15]

    # Val
    X_val_num = X_val[:, :12]
    X_val_seq = prepare_lstm_sequences(X_val_num)
    X_val_curr = X_val_num[:, :2]
    X_val_state = X_val[:, 12:13]
    X_val_dist = X_val[:, 13:14]
    X_val_crop = X_val[:, 14:15]

    # Test
    X_test_num = X_test[:, :12]
    X_test_seq = prepare_lstm_sequences(X_test_num)
    X_test_curr = X_test_num[:, :2]
    X_test_state = X_test[:, 12:13]
    X_test_dist = X_test[:, 13:14]
    X_test_crop = X_test[:, 14:15]

    # 3. Option B: Bidirectional LSTM Pipeline
    print("\n[3/4] Running Option B (Bidirectional LSTM)...")
    bilstm_nn_preds, bilstm_xgb_preds = train_lstm_pipeline(
        bidirectional=True, epochs=epochs, batch_size=batch_size, n_estimators=n_estimators,
        learning_rate=learning_rate,
        X_train_seq=X_train_seq, X_train_curr=X_train_curr, X_train_state=X_train_state, X_train_dist=X_train_dist, X_train_crop=X_train_crop, y_train=y_train,
        X_val_seq=X_val_seq, X_val_curr=X_val_curr, X_val_state=X_val_state, X_val_dist=X_val_dist, X_val_crop=X_val_crop, y_val=y_val,
        X_test_seq=X_test_seq, X_test_curr=X_test_curr, X_test_state=X_test_state, X_test_dist=X_test_dist, X_test_crop=X_test_crop, y_test=y_test,
        num_states=num_states, num_districts=num_districts, num_crops=num_crops
    )

    # 4. Option C: Unidirectional LSTM Pipeline
    print("\n[4/4] Running Option C (Unidirectional LSTM)...")
    unilstm_nn_preds, unilstm_xgb_preds = train_lstm_pipeline(
        bidirectional=False, epochs=epochs, batch_size=batch_size, n_estimators=n_estimators,
        learning_rate=learning_rate,
        X_train_seq=X_train_seq, X_train_curr=X_train_curr, X_train_state=X_train_state, X_train_dist=X_train_dist, X_train_crop=X_train_crop, y_train=y_train,
        X_val_seq=X_val_seq, X_val_curr=X_val_curr, X_val_state=X_val_state, X_val_dist=X_val_dist, X_val_crop=X_val_crop, y_val=y_val,
        X_test_seq=X_test_seq, X_test_curr=X_test_curr, X_test_state=X_test_state, X_test_dist=X_test_dist, X_test_crop=X_test_crop, y_test=y_test,
        num_states=num_states, num_districts=num_districts, num_crops=num_crops
    )

    # Save intermediate predictions
    np.savez(os.path.join(MODEL_DIR, 'lstm_predictions.npz'),
             bilstm_nn_preds=bilstm_nn_preds,
             bilstm_xgb_preds=bilstm_xgb_preds,
             unilstm_nn_preds=unilstm_nn_preds,
             unilstm_xgb_preds=unilstm_xgb_preds)

    print("\n" + "="*80)
    print("  OPTION B & C TRAINING COMPLETE")
    print(f"  All hybrid models saved in: {MODEL_DIR}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    run_training_lstm_hybrids(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_estimators=args.n_estimators,
        learning_rate=args.lr
    )
