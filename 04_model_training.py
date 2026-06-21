"""
=============================================================================
Module 04: Model Training — 1D CNN-Recursive BiLSTM Hybrid
=============================================================================
Implements the proposed hybrid deep learning architecture:
  1. 1D Convolutional Neural Network (CNN) for local pattern extraction
  2. Recursive Bidirectional LSTM (BiLSTM) for temporal dependencies
  3. Combined hybrid model: 1D CNN → Recursive BiLSTM → Dense output
  4. Comparison baseline models (standalone CNN, standalone BiLSTM,
     Random Forest, Gradient Boosting) for benchmarking
  5. Training with early stopping, learning rate scheduling, and checkpoints
=============================================================================
"""

import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = os.path.join(BASE_DIR, 'feature_selection_output')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOT_DIR = os.path.join(MODEL_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── GPU Configuration ────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  GPU(s) detected: {len(gpus)}")
    except RuntimeError as e:
        print(f"  GPU config error: {e}")
else:
    print("  No GPU detected — using CPU")


# ── Data Preparation ─────────────────────────────────────────────────────

def prepare_data_for_model(X, reshape_for_cnn=True):
    """
    Reshape data for 1D CNN / BiLSTM input.
    Input shape: (samples, features) → (samples, features, 1) for Conv1D
    or (samples, timesteps, features) for LSTM.
    """
    if reshape_for_cnn:
        return X.reshape(X.shape[0], X.shape[1], 1)
    return X


# ── Model Architectures ─────────────────────────────────────────────────

def build_1d_cnn(input_shape, learning_rate=0.001):
    """
    Build standalone 1D CNN model.
    Architecture: Conv1D → BatchNorm → MaxPool → Conv1D → GlobalAvgPool → Dense
    """
    inputs = layers.Input(shape=input_shape, name='cnn_input')
    
    # Block 1
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 2
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu',
                      kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, name='yield_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='1D_CNN')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_bilstm(input_shape, learning_rate=0.001):
    """
    Build standalone Recursive Bidirectional LSTM model.
    Architecture: BiLSTM → BiLSTM (recursive/stacked) → Dense
    """
    inputs = layers.Input(shape=input_shape, name='bilstm_input')
    
    # Recursive BiLSTM layers (stacked)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2,
                     recurrent_dropout=0.1),
        name='bilstm_1')(inputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2,
                     recurrent_dropout=0.1),
        name='bilstm_2')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False, dropout=0.2,
                     recurrent_dropout=0.1),
        name='bilstm_3')(x)
    x = layers.BatchNormalization()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu',
                      kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, name='yield_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Recursive_BiLSTM')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_cnn_bilstm_hybrid(input_shape, learning_rate=0.001):
    """
    Build the proposed 1D CNN-Recursive BiLSTM hybrid model.
    
    Architecture (per the diagram):
      Input → 1D CNN blocks (local feature extraction)
            → Recursive BiLSTM (sequential dependency capture)
            → Dense layers → Yield Prediction
    """
    inputs = layers.Input(shape=input_shape, name='hybrid_input')
    
    # ── 1D CNN Branch: Local pattern extraction ──
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001),
                       name='conv1d_1')(inputs)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same', name='maxpool_1')(x)
    x = layers.Dropout(0.2, name='dropout_conv1')(x)
    
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001),
                       name='conv1d_2')(x)
    x = layers.BatchNormalization(name='bn_conv2')(x)
    x = layers.Dropout(0.2, name='dropout_conv2')(x)
    
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001),
                       name='conv1d_3')(x)
    x = layers.BatchNormalization(name='bn_conv3')(x)
    
    # ── Recursive BiLSTM Branch: Sequential dependencies ──
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2,
                     recurrent_dropout=0.1),
        name='bilstm_layer_1')(x)
    x = layers.BatchNormalization(name='bn_lstm1')(x)
    
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2,
                     recurrent_dropout=0.1),
        name='bilstm_layer_2')(x)
    x = layers.BatchNormalization(name='bn_lstm2')(x)
    
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False, dropout=0.2,
                     recurrent_dropout=0.1),
        name='bilstm_layer_3')(x)
    x = layers.BatchNormalization(name='bn_lstm3')(x)
    
    # ── Fully Connected Regression Head ──
    x = layers.Dense(256, activation='relu',
                      kernel_regularizer=keras.regularizers.l2(0.001),
                      name='dense_1')(x)
    x = layers.Dropout(0.4, name='dropout_dense1')(x)
    
    x = layers.Dense(128, activation='relu',
                      kernel_regularizer=keras.regularizers.l2(0.001),
                      name='dense_2')(x)
    x = layers.Dropout(0.3, name='dropout_dense2')(x)
    
    x = layers.Dense(64, activation='relu', name='dense_3')(x)
    x = layers.Dropout(0.2, name='dropout_dense3')(x)
    
    outputs = layers.Dense(1, name='yield_prediction')(x)
    
    model = Model(inputs=inputs, outputs=outputs, 
                  name='1D_CNN_Recursive_BiLSTM')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ── Training Utilities ────────────────────────────────────────────────────

def get_callbacks(model_name, patience=15):
    """Create training callbacks: early stopping, LR scheduler, checkpoints."""
    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f'{model_name}_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
    ]
    return cb_list


def train_deep_model(model, X_train, y_train, X_val, y_val,
                      epochs=200, batch_size=64):
    """Train a Keras model with callbacks."""
    model_name = model.name
    print(f"\n  Training {model_name}...")
    print(f"    Input shape:  {X_train.shape}")
    print(f"    Parameters:   {model.count_params():,}")
    
    model.summary(print_fn=lambda x: None)  # Suppress console summary
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(model_name),
        verbose=1
    )
    
    return history


def train_ml_baselines(X_train, y_train, X_val, y_val):
    """Train traditional ML baseline models for comparison."""
    print("\n  [Training ML Baselines]")
    
    baselines = {}
    
    # Random Forest
    print("    Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
    rf_r2 = r2_score(y_val, rf_pred)
    baselines['RandomForest'] = {'model': rf, 'rmse': rf_rmse, 'r2': rf_r2}
    print(f"      RMSE: {rf_rmse:.4f} | R²: {rf_r2:.4f}")
    
    # Gradient Boosting
    print("    Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_val)
    gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))
    gb_r2 = r2_score(y_val, gb_pred)
    baselines['GradientBoosting'] = {'model': gb, 'rmse': gb_rmse, 'r2': gb_r2}
    print(f"      RMSE: {gb_rmse:.4f} | R²: {gb_r2:.4f}")
    
    return baselines


# ── Visualisation ─────────────────────────────────────────────────────────

def plot_training_history(history, model_name, save_dir):
    """Plot training and validation loss/MAE curves."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title(f'{model_name} — Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_title(f'{model_name} — MAE', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Training History — {model_name}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'training_history_{model_name}.png'))
    plt.close(fig)
    print(f"    ✓ Saved: training_history_{model_name}.png")


# ── Main Execution ────────────────────────────────────────────────────────

def run_training(epochs=200, batch_size=64, learning_rate=0.001):
    """Execute the complete model training pipeline."""
    print("="*80)
    print("  CROP YIELD PREDICTION — MODEL TRAINING")
    print("  Module 04: 1D CNN-Recursive BiLSTM Hybrid")
    print("="*80)
    
    # 1. Load selected features data
    print("\n[1/6] Loading feature-selected data...")
    data = np.load(os.path.join(FEATURE_DIR, 'selected_features_data.npz'))
    X_train = data['X_train']
    y_train = data['y_train']
    X_val   = data['X_val']
    y_val   = data['y_val']
    X_test  = data['X_test']
    y_test  = data['y_test']
    
    print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}   | y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    
    n_features = X_train.shape[1]
    input_shape_3d = (n_features, 1)  # For Conv1D/LSTM: (timesteps, channels)
    
    # Reshape for deep learning models
    X_train_3d = prepare_data_for_model(X_train)
    X_val_3d   = prepare_data_for_model(X_val)
    X_test_3d  = prepare_data_for_model(X_test)
    
    trained_models = {}
    histories = {}
    
    # 2. Train standalone 1D CNN
    print("\n[2/6] Training standalone 1D CNN...")
    cnn_model = build_1d_cnn(input_shape_3d, learning_rate)
    cnn_history = train_deep_model(
        cnn_model, X_train_3d, y_train, X_val_3d, y_val,
        epochs=epochs, batch_size=batch_size)
    trained_models['1D_CNN'] = cnn_model
    histories['1D_CNN'] = cnn_history
    plot_training_history(cnn_history, '1D_CNN', PLOT_DIR)
    
    # 3. Train standalone BiLSTM
    print("\n[3/6] Training standalone Recursive BiLSTM...")
    bilstm_model = build_bilstm(input_shape_3d, learning_rate)
    bilstm_history = train_deep_model(
        bilstm_model, X_train_3d, y_train, X_val_3d, y_val,
        epochs=epochs, batch_size=batch_size)
    trained_models['Recursive_BiLSTM'] = bilstm_model
    histories['Recursive_BiLSTM'] = bilstm_history
    plot_training_history(bilstm_history, 'Recursive_BiLSTM', PLOT_DIR)
    
    # 4. Train PROPOSED HYBRID: 1D CNN-Recursive BiLSTM
    print("\n[4/6] Training PROPOSED HYBRID: 1D CNN-Recursive BiLSTM...")
    hybrid_model = build_cnn_bilstm_hybrid(input_shape_3d, learning_rate)
    hybrid_history = train_deep_model(
        hybrid_model, X_train_3d, y_train, X_val_3d, y_val,
        epochs=epochs, batch_size=batch_size)
    trained_models['1D_CNN_Recursive_BiLSTM'] = hybrid_model
    histories['1D_CNN_Recursive_BiLSTM'] = hybrid_history
    plot_training_history(hybrid_history, '1D_CNN_Recursive_BiLSTM', PLOT_DIR)
    
    # 5. Train ML Baselines
    print("\n[5/6] Training ML baseline models...")
    baselines = train_ml_baselines(X_train, y_train, X_val, y_val)
    trained_models.update({name: info['model'] for name, info in baselines.items()})
    
    # 6. Save all models
    print("\n[6/6] Saving models...")
    
    for name, model in trained_models.items():
        if isinstance(model, keras.Model):
            model.save(os.path.join(MODEL_DIR, f'{name}.keras'))
            print(f"    ✓ {name}.keras")
        else:
            with open(os.path.join(MODEL_DIR, f'{name}.pkl'), 'wb') as f:
                pickle.dump(model, f)
            print(f"    ✓ {name}.pkl")
    
    # Save test data for evaluation
    np.savez(os.path.join(MODEL_DIR, 'test_data.npz'),
             X_test=X_test, y_test=y_test,
             X_test_3d=X_test_3d)
    
    print("\n" + "="*80)
    print("  MODEL TRAINING COMPLETE")
    print(f"  Models trained: {list(trained_models.keys())}")
    print(f"  Artifacts saved to: {MODEL_DIR}")
    print("="*80)
    
    return trained_models, histories


if __name__ == '__main__':
    run_training(epochs=200, batch_size=64)