"""
=============================================================================
Module 05: Comprehensive Model Evaluation
=============================================================================
Evaluates all trained models using the full set of metrics specified:
  Regression:   RMSE, MSE, MAE, MBE (Mean Bias Error), R-squared
  Classification-style (binned): Accuracy, F1-Score, Precision, Recall
  
Generates:
  - Metric comparison tables (all models)
  - Actual vs Predicted scatter plots
  - Residual analysis plots
  - Error distribution plots
  - Model ranking summary
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FEATURE_DIR = os.path.join(BASE_DIR, 'feature_selection_output')
EVAL_DIR = os.path.join(BASE_DIR, 'evaluation_output')
PLOT_DIR = os.path.join(EVAL_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Plot styling
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'figure.dpi': 150,
    'font.size': 12,
    'figure.facecolor': 'white',
})
sns.set_theme(style='whitegrid', palette='muted')


# ── Regression Metrics ────────────────────────────────────────────────────

def compute_regression_metrics(y_true, y_pred):
    """
    Compute all regression evaluation metrics.
    
    Returns dict with: RMSE, MSE, MAE, MBE, R²
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mbe = np.mean(y_pred - y_true)  # Mean Bias Error
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'MBE': mbe,
        'R²': r2,
    }


# ── Classification-style Metrics (Binned Yield) ──────────────────────────

def compute_classification_metrics(y_true, y_pred, n_bins=5):
    """
    Convert continuous yield predictions to bins for classification metrics.
    Uses quantile-based binning on true values.
    
    Returns dict with: Accuracy, F1-Score, Precision, Recall
    """
    # Create quantile-based bins from true values
    bin_edges = np.quantile(y_true, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    y_true_binned = np.digitize(y_true, bin_edges) - 1
    y_pred_binned = np.digitize(y_pred, bin_edges) - 1
    
    # Clip to valid range
    y_true_binned = np.clip(y_true_binned, 0, n_bins - 1)
    y_pred_binned = np.clip(y_pred_binned, 0, n_bins - 1)
    
    accuracy = accuracy_score(y_true_binned, y_pred_binned)
    f1 = f1_score(y_true_binned, y_pred_binned, average='weighted', zero_division=0)
    precision = precision_score(y_true_binned, y_pred_binned, average='weighted', zero_division=0)
    recall = recall_score(y_true_binned, y_pred_binned, average='weighted', zero_division=0)
    
    return {
        'Accuracy': accuracy,
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall,
    }


# ── Model Loading & Prediction ───────────────────────────────────────────

def load_all_models():
    """Load all trained models."""
    models = {}
    
    # Deep learning models
    dl_model_names = ['1D_CNN', 'Recursive_BiLSTM', '1D_CNN_Recursive_BiLSTM']
    for name in dl_model_names:
        path = os.path.join(MODEL_DIR, f'{name}.keras')
        if os.path.exists(path):
            models[name] = keras.models.load_model(path)
            print(f"  ✓ Loaded: {name}")
        else:
            # Try with best suffix
            best_path = os.path.join(MODEL_DIR, f'{name}_best.keras')
            if os.path.exists(best_path):
                models[name] = keras.models.load_model(best_path)
                print(f"  ✓ Loaded: {name} (best)")
    
    # ML models
    ml_model_names = ['RandomForest', 'GradientBoosting']
    for name in ml_model_names:
        path = os.path.join(MODEL_DIR, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"  ✓ Loaded: {name}")
    
    return models


def get_predictions(models, X_test, X_test_3d):
    """Get predictions from all models."""
    predictions = {}
    
    for name, model in models.items():
        try:
            if isinstance(model, keras.Model):
                pred = model.predict(X_test_3d, verbose=0).flatten()
            else:
                pred = model.predict(X_test)
            predictions[name] = pred
            print(f"  ✓ Predicted: {name} ({len(pred):,} samples)")
        except Exception as e:
            print(f"  ✗ Failed: {name} — {e}")
    
    return predictions


# ── Comprehensive Evaluation ──────────────────────────────────────────────

def evaluate_all_models(y_test, predictions):
    """Evaluate all models and create comparison table."""
    print("\n" + "="*80)
    print("  MODEL EVALUATION RESULTS")
    print("="*80)
    
    results = []
    
    for name, y_pred in predictions.items():
        # Regression metrics
        reg_metrics = compute_regression_metrics(y_test, y_pred)
        
        # Classification-style metrics (binned)
        cls_metrics = compute_classification_metrics(y_test, y_pred)
        
        # Combine
        combined = {'Model': name}
        combined.update(reg_metrics)
        combined.update(cls_metrics)
        results.append(combined)
        
        print(f"\n  ── {name} ──")
        print(f"    Regression:       RMSE={reg_metrics['RMSE']:.4f}  "
              f"MSE={reg_metrics['MSE']:.4f}  MAE={reg_metrics['MAE']:.4f}  "
              f"MBE={reg_metrics['MBE']:.4f}  R²={reg_metrics['R²']:.4f}")
        print(f"    Classification:   Accuracy={cls_metrics['Accuracy']:.4f}  "
              f"F1={cls_metrics['F1-Score']:.4f}  "
              f"Precision={cls_metrics['Precision']:.4f}  "
              f"Recall={cls_metrics['Recall']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    # Sort by RMSE (lower is better)
    results_df = results_df.sort_values('RMSE')
    
    print("\n" + "─"*80)
    print("  COMPARISON TABLE (Sorted by RMSE — lower is better)")
    print("─"*80)
    print(results_df.to_string(index=False))
    
    return results_df


# ── Visualisation ─────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_test, predictions, save_dir):
    """Plot actual vs predicted scatter for each model."""
    n_models = len(predictions)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 6*rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = sns.color_palette('Set2', n_models)
    
    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color=colors[i], label=name)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                label='Perfect prediction')
        
        # R² annotation
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax.set_xlabel('Actual Yield (hg/ha)', fontweight='bold')
        ax.set_ylabel('Predicted Yield (hg/ha)', fontweight='bold')
        ax.set_title(f'{name}', fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
    
    # Hide unused axes
    for j in range(len(predictions), len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Actual vs Predicted — All Models', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'actual_vs_predicted.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: actual_vs_predicted.png")


def plot_residual_analysis(y_test, predictions, save_dir):
    """Plot residual distributions for each model."""
    n_models = len(predictions)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    colors = sns.color_palette('husl', n_models)
    
    for i, (name, y_pred) in enumerate(predictions.items()):
        residuals = y_test - y_pred
        
        # Residual scatter
        axes[0, i].scatter(y_pred, residuals, alpha=0.3, s=8, color=colors[i])
        axes[0, i].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        axes[0, i].set_xlabel('Predicted', fontweight='bold')
        axes[0, i].set_ylabel('Residual', fontweight='bold')
        axes[0, i].set_title(f'{name}\nResiduals vs Predicted', fontweight='bold')
        
        # Residual histogram
        axes[1, i].hist(residuals, bins=50, color=colors[i], edgecolor='white', alpha=0.8)
        axes[1, i].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        axes[1, i].set_xlabel('Residual', fontweight='bold')
        axes[1, i].set_ylabel('Frequency', fontweight='bold')
        axes[1, i].set_title(f'{name}\nResidual Distribution', fontweight='bold')
        
        # Annotate stats
        axes[1, i].text(0.95, 0.95,
                         f'Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}',
                         transform=axes[1, i].transAxes, fontsize=10,
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle('Residual Analysis — All Models', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'residual_analysis.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: residual_analysis.png")


def plot_metric_comparison(results_df, save_dir):
    """Plot bar charts comparing all metrics across models."""
    regression_metrics = ['RMSE', 'MSE', 'MAE', 'R²']
    classification_metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    colors = sns.color_palette('Set2', len(results_df))
    
    # Regression metrics
    for i, metric in enumerate(regression_metrics):
        ax = axes[0, i]
        bars = ax.bar(results_df['Model'], results_df[metric], 
                       color=colors, edgecolor='white', alpha=0.85)
        ax.set_title(metric, fontweight='bold', fontsize=14)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # Annotate values
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Classification metrics
    for i, metric in enumerate(classification_metrics):
        ax = axes[1, i]
        bars = ax.bar(results_df['Model'], results_df[metric],
                       color=colors, edgecolor='white', alpha=0.85)
        ax.set_title(metric, fontweight='bold', fontsize=14)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Model Performance Comparison — All Metrics', 
                 fontsize=18, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'metric_comparison.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: metric_comparison.png")


def plot_error_distribution_comparison(y_test, predictions, save_dir):
    """Overlay error distributions for all models."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = sns.color_palette('husl', len(predictions))
    
    for i, (name, y_pred) in enumerate(predictions.items()):
        errors = np.abs(y_test - y_pred)
        ax.hist(errors, bins=50, alpha=0.4, color=colors[i], label=name,
                edgecolor='white')
    
    ax.set_xlabel('Absolute Error', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Absolute Error Distribution — All Models', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'error_distributions.png'))
    plt.close(fig)
    print(f"  ✓ Saved: error_distributions.png")


def generate_evaluation_report(results_df, y_test, predictions, save_dir):
    """Generate a detailed text evaluation report."""
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("  CROP YIELD PREDICTION — MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test samples: {len(y_test):,}\n")
        f.write(f"Models evaluated: {len(predictions)}\n\n")
        
        # Full comparison table
        f.write("─"*80 + "\n")
        f.write("PERFORMANCE COMPARISON\n")
        f.write("─"*80 + "\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        
        # Best model
        best_model = results_df.iloc[0]['Model']
        f.write(f"BEST MODEL (by RMSE): {best_model}\n")
        f.write(f"  RMSE:      {results_df.iloc[0]['RMSE']:.4f}\n")
        f.write(f"  R²:        {results_df.iloc[0]['R²']:.4f}\n")
        f.write(f"  Accuracy:  {results_df.iloc[0]['Accuracy']:.4f}\n")
        f.write(f"  F1-Score:  {results_df.iloc[0]['F1-Score']:.4f}\n\n")
        
        # Proposed model performance
        if '1D_CNN_Recursive_BiLSTM' in predictions:
            proposed = results_df[results_df['Model'] == '1D_CNN_Recursive_BiLSTM']
            if not proposed.empty:
                f.write("─"*60 + "\n")
                f.write("PROPOSED MODEL: 1D CNN-Recursive BiLSTM\n")
                f.write("─"*60 + "\n")
                for col in proposed.columns[1:]:
                    f.write(f"  {col}: {proposed.iloc[0][col]:.6f}\n")
        
        # Per-model detailed analysis
        f.write("\n" + "─"*60 + "\n")
        f.write("DETAILED PER-MODEL ANALYSIS\n")
        f.write("─"*60 + "\n")
        
        for name, y_pred in predictions.items():
            residuals = y_test - y_pred
            f.write(f"\n  [{name}]\n")
            f.write(f"    Residual Stats:\n")
            f.write(f"      Mean:   {residuals.mean():.4f}\n")
            f.write(f"      Std:    {residuals.std():.4f}\n")
            f.write(f"      Min:    {residuals.min():.4f}\n")
            f.write(f"      Max:    {residuals.max():.4f}\n")
            f.write(f"      Median: {np.median(residuals):.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("  END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"  ✓ Report saved: {report_path}")
    
    # Also save results as CSV
    csv_path = os.path.join(save_dir, 'evaluation_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  ✓ CSV saved: {csv_path}")


def generate_prediction_samples(y_test, predictions, X_test, save_dir, n_samples=30):
    """
    Generate a sample of yield predictions with decoded labels for display.
    Shows actual vs predicted yield for the proposed model and best baseline.
    """
    preprocess_dir = os.path.join(BASE_DIR, 'preprocessed_data')
    
    try:
        with open(os.path.join(preprocess_dir, 'label_encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
        with open(os.path.join(preprocess_dir, 'scaler.pkl'), 'rb') as f:
            scaler_data = pickle.load(f)
        with open(os.path.join(preprocess_dir, 'feature_columns.pkl'), 'rb') as f:
            col_info = pickle.load(f)
    except Exception as e:
        print(f"  ⚠ Could not load encoders/scaler: {e}")
        return
    
    scaler = scaler_data['scaler']
    feature_cols = col_info['feature_cols']
    scaled_cols = scaler_data['columns']
    
    # Load the PRE-selection test data (has all 6 original features)
    pre_splits = np.load(os.path.join(preprocess_dir, 'train_test_splits.npz'))
    X_test_full = pre_splits['X_test']  # shape: (5187, 6)
    
    # Build full array matching scaler columns for inverse transform
    n = len(y_test)
    full_scaled = np.zeros((n, len(scaled_cols)))
    
    # Map all 6 original feature columns into their scaler positions
    for i, fc in enumerate(feature_cols):
        if fc in scaled_cols:
            j = scaled_cols.index(fc)
            full_scaled[:, j] = X_test_full[:, i]
    
    # Put y_test in target column position
    target_idx = scaled_cols.index('hg/ha_yield')
    full_scaled[:, target_idx] = y_test
    
    # Inverse transform to original scale
    full_original = scaler.inverse_transform(full_scaled)
    
    y_test_original = full_original[:, target_idx]
    
    # Decode Area and Item
    area_codes = np.round(full_original[:, scaled_cols.index('Area_encoded')]).astype(int)
    item_codes = np.round(full_original[:, scaled_cols.index('Item_encoded')]).astype(int)
    
    area_codes = np.clip(area_codes, 0, len(encoders['Area'].classes_) - 1)
    item_codes = np.clip(item_codes, 0, len(encoders['Item'].classes_) - 1)
    
    areas = encoders['Area'].inverse_transform(area_codes)
    items = encoders['Item'].inverse_transform(item_codes)
    years = np.round(full_original[:, scaled_cols.index('Year')]).astype(int)
    
    # Pick diverse samples - stratified across different crops
    rng = np.random.RandomState(42)
    indices = rng.choice(len(y_test), min(n_samples * 3, len(y_test)), replace=False)
    
    seen_crops = {}
    selected = []
    for idx in indices:
        crop = items[idx]
        if crop not in seen_crops:
            seen_crops[crop] = 0
        if seen_crops[crop] < 3:
            selected.append(idx)
            seen_crops[crop] += 1
        if len(selected) >= n_samples:
            break
    
    if len(selected) < n_samples:
        remaining = [i for i in indices if i not in selected]
        selected.extend(remaining[:n_samples - len(selected)])
    
    rows = []
    for idx in selected:
        row = {
            'Country': areas[idx],
            'Crop': items[idx],
            'Year': int(years[idx]),
            'Actual_Yield': round(float(y_test_original[idx]), 1),
        }
        
        # Inverse transform each model's prediction
        for model_name, preds in predictions.items():
            pred_full = full_scaled[idx].copy()
            pred_full[target_idx] = preds[idx]
            pred_original = scaler.inverse_transform(pred_full.reshape(1, -1))[0, target_idx]
            row[f'Pred_{model_name}'] = round(float(pred_original), 1)
        
        rows.append(row)
    
    pred_df = pd.DataFrame(rows)
    pred_path = os.path.join(save_dir, 'prediction_samples.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"  ✓ Prediction samples saved: {pred_path}")


# ── Main Execution ────────────────────────────────────────────────────────

def run_evaluation():
    """Execute the complete evaluation pipeline."""
    print("="*80)
    print("  CROP YIELD PREDICTION — MODEL EVALUATION")
    print("  Module 05: RMSE, MSE, MAE, MBE, R², Accuracy, F1, Precision, Recall")
    print("="*80)
    
    # 1. Load test data
    print("\n[1/6] Loading test data...")
    test_data = np.load(os.path.join(MODEL_DIR, 'test_data.npz'))
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    X_test_3d = test_data['X_test_3d']
    print(f"  Test samples: {len(y_test):,}")
    
    # 2. Load models
    print("\n[2/6] Loading trained models...")
    models = load_all_models()
    
    if not models:
        print("  ✗ No models found! Run 04_model_training.py first.")
        return
    
    # 3. Get predictions
    print("\n[3/6] Generating predictions...")
    predictions = get_predictions(models, X_test, X_test_3d)
    
    # 4. Evaluate
    print("\n[4/6] Computing evaluation metrics...")
    results_df = evaluate_all_models(y_test, predictions)
    
    # 5. Generate visualisations and report
    print("\n[5/6] Generating visualisations and report...")
    plot_actual_vs_predicted(y_test, predictions, PLOT_DIR)
    plot_residual_analysis(y_test, predictions, PLOT_DIR)
    plot_metric_comparison(results_df, PLOT_DIR)
    plot_error_distribution_comparison(y_test, predictions, PLOT_DIR)
    generate_evaluation_report(results_df, y_test, predictions, EVAL_DIR)
    
    # 6. Generate prediction samples
    print("\n[6/6] Generating yield prediction samples...")
    generate_prediction_samples(y_test, predictions, X_test, EVAL_DIR)
    
    print("\n" + "="*80)
    print("  EVALUATION COMPLETE")
    print(f"  Best model: {results_df.iloc[0]['Model']} "
          f"(RMSE={results_df.iloc[0]['RMSE']:.4f}, R²={results_df.iloc[0]['R²']:.4f})")
    print(f"  Artifacts saved to: {EVAL_DIR}")
    print("="*80)
    
    return results_df, predictions


if __name__ == '__main__':
    run_evaluation()

