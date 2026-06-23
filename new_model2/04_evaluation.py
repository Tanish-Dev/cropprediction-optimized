"""
=============================================================================
Stage 4: Comprehensive Model Evaluation Script (Old Dataset)
=============================================================================
Computes regression metrics (RMSE, MSE, MAE, MBE, R²) for all candidates:
  Option A: RF, XGBoost, MLP, Stacking Hybrid
  Option B: Bidirectional LSTM Neural Net, BiLSTM-XGBoost Hybrid
  Option C: Unidirectional LSTM Neural Net, UniLSTM-XGBoost Hybrid
Saves report, CSV metrics, prediction samples, and comparison plots.
=============================================================================
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(BASE_DIR, 'preprocessed_data')
MODEL_DIR        = os.path.join(BASE_DIR, 'models')
EVAL_DIR         = os.path.join(BASE_DIR, 'evaluation_output')
PLOT_DIR         = os.path.join(EVAL_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Styling
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'figure.dpi': 150,
    'font.size': 12,
    'figure.facecolor': 'white',
})
sns.set_theme(style='whitegrid', palette='muted')


def compute_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    mbe  = np.mean(y_pred - y_true)
    r2   = r2_score(y_true, y_pred)

    y_true_safe = np.where(y_true == 0, 1e-5, y_true)
    mape        = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    accuracy    = max(0.0, 100.0 - mape)

    acc_10 = np.mean(np.abs(y_true - y_pred) / y_true_safe <= 0.10) * 100
    acc_20 = np.mean(np.abs(y_true - y_pred) / y_true_safe <= 0.20) * 100

    return {
        'RMSE':       rmse,
        'MSE':        mse,
        'MAE':        mae,
        'MBE':        mbe,
        'R²':         r2,
        'MAPE':       mape,
        'Accuracy %': accuracy,
        'Acc_10%':    acc_10,
        'Acc_20%':    acc_20
    }


def run_evaluation():
    print("="*80)
    print("  CROP YIELD PREDICTION (OLD DATASET) — COMPREHENSIVE EVALUATION")
    print("  Stage 4: Compute Metrics, Generate Reports & Plots")
    print("="*80)

    # 1. Load test data and scalers
    splits     = np.load(os.path.join(PREPROCESSED_DIR, 'train_test_splits.npz'))
    y_test_raw = splits['y_test_raw']
    X_test     = splits['X_test']

    with open(os.path.join(PREPROCESSED_DIR, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    target_scaler = scalers['target_scaler']

    # 2. Load prediction artifacts
    pred_opt_a = np.load(os.path.join(MODEL_DIR, 'option_a_predictions.npz'))
    pred_lstm  = np.load(os.path.join(MODEL_DIR, 'lstm_predictions.npz'))

    raw_preds = {
        'Option A: Random Forest':      pred_opt_a['rf_preds'],
        'Option A: XGBoost':            pred_opt_a['xgb_preds'],
        'Option A: MLP Embeddings':     pred_opt_a['mlp_preds'],
        'Option A: Stacking Hybrid':    pred_opt_a['stacking_preds'],

        'Option B: BiLSTM Neural Net':      pred_lstm['bilstm_nn_preds'],
        'Option B: BiLSTM-XGBoost Hybrid':  pred_lstm['bilstm_xgb_preds'],

        'Option C: UniLSTM Neural Net':     pred_lstm['unilstm_nn_preds'],
        'Option C: UniLSTM-XGBoost Hybrid': pred_lstm['unilstm_xgb_preds']
    }

    # Inverse-scale predictions back to original hg/ha scale
    unscaled_preds = {}
    print("\n[1/3] Inverse scaling predictions back to hg/ha units...")
    for name, pred in raw_preds.items():
        unscaled = target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        unscaled_preds[name] = unscaled
        print(f"    ✓ Processed: {name}")

    # 3. Compute Metrics
    print("\n[2/3] Computing regression metrics against true yield...")
    results = []
    for name, pred in unscaled_preds.items():
        metrics = compute_metrics(y_test_raw, pred)
        entry = {'Model': name}
        entry.update(metrics)
        results.append(entry)

    results_df = pd.DataFrame(results).sort_values(by='RMSE')
    print("\n" + "─"*80)
    print("  COMPARISON SUMMARY (Sorted by RMSE)")
    print("─"*80)
    print(results_df.to_string(index=False))

    # Save CSV
    results_df.to_csv(os.path.join(EVAL_DIR, 'evaluation_results.csv'), index=False)

    # 4. Generate report file
    report_path = os.path.join(EVAL_DIR, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("  CROP YIELD PREDICTION — EXPERIMENT COMPARISON REPORT (OLD DATASET)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Set Size: {len(y_test_raw):,} samples (2011 - 2013)\n\n")

        f.write("─"*80 + "\n")
        f.write("PERFORMANCE SUMMARY TABLE\n")
        f.write("─"*80 + "\n")
        f.write(results_df.to_string(index=False) + "\n\n")

        best_model = results_df.iloc[0]['Model']
        best_rmse  = results_df.iloc[0]['RMSE']
        best_r2    = results_df.iloc[0]['R²']

        f.write(f"BEST PERFORMING MODEL: {best_model}\n")
        f.write(f"  RMSE: {best_rmse:.2f} hg/ha\n")
        f.write(f"  R²:   {best_r2:.4f}\n\n")

        f.write("─"*80 + "\n")
        f.write("DETAILED MODEL EVALUATION ANALYSIS\n")
        f.write("─"*80 + "\n")

        for name, pred in unscaled_preds.items():
            residuals = y_test_raw - pred
            f.write(f"\n  [{name}]\n")
            f.write(f"    RMSE (Error):       {np.sqrt(np.mean(residuals**2)):.2f} hg/ha\n")
            f.write(f"    MAE (Abs Error):    {np.mean(np.abs(residuals)):.2f} hg/ha\n")
            f.write(f"    MBE (Bias):         {np.mean(residuals):.2f} hg/ha\n")
            f.write(f"    R² Score:           {r2_score(y_test_raw, pred):.4f}\n")
            f.write(f"    Residual Stats:\n")
            f.write(f"      Std Dev:          {residuals.std():.2f}\n")
            f.write(f"      Min / Max Error:  {residuals.min():.2f} / {residuals.max():.2f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("  END OF REPORT\n")
        f.write("="*80 + "\n")
    print(f"    ✓ Saved report: {report_path}")

    # 5. Generate prediction samples CSV
    print("\n[3/3] Generating prediction samples CSV...")
    with open(os.path.join(PREPROCESSED_DIR, 'encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)

    NUM_NUMERIC = X_test.shape[1] - 2
    country_enc = X_test[:, NUM_NUMERIC].astype(int)
    crop_enc    = X_test[:, NUM_NUMERIC+1].astype(int)

    feature_scaler = scalers['feature_scaler']
    dummy_orig     = feature_scaler.inverse_transform(X_test[:, :NUM_NUMERIC])

    years = np.round(dummy_orig[:, 0]).astype(int)

    countries = encoders['Area'].inverse_transform(country_enc)
    crops     = encoders['Item'].inverse_transform(crop_enc)

    samples_df = pd.DataFrame({
        'Country':              countries,
        'Crop':                 crops,
        'Year':                 years,
        'Actual_Yield_hg_ha':   np.round(y_test_raw, 1)
    })

    for name, pred in unscaled_preds.items():
        col = (name.replace('Option A: ', 'A_')
                   .replace('Option B: ', 'B_')
                   .replace('Option C: ', 'C_')
                   .replace(' ', '_')
                   .replace(':', ''))
        samples_df[col] = np.round(pred, 1)

    rng           = np.random.RandomState(42)
    sample_indices = rng.choice(len(samples_df), min(100, len(samples_df)), replace=False)
    sample_to_save = samples_df.iloc[sample_indices].copy()
    sample_to_save.to_csv(os.path.join(EVAL_DIR, 'prediction_samples.csv'), index=False)
    print(f"    ✓ Saved: prediction_samples.csv")

    # 6. Generate Comparative Visualizations
    print("  Generating comparative plots...")

    # Actual vs Predicted scatter for top 3 models
    top_models = results_df['Model'].head(3).tolist()
    fig, axes  = plt.subplots(1, 3, figsize=(20, 6))
    colors     = sns.color_palette('Set2', 3)

    for i, name in enumerate(top_models):
        ax   = axes[i]
        pred = unscaled_preds[name]

        ax.scatter(y_test_raw, pred, alpha=0.3, s=8, color=colors[i])
        min_val = min(y_test_raw.min(), pred.min())
        max_val = max(y_test_raw.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect prediction')

        r2_val   = results_df[results_df['Model'] == name]['R²'].values[0]
        rmse_val = results_df[results_df['Model'] == name]['RMSE'].values[0]
        ax.set_title(f"{name}\nR²={r2_val:.4f} | RMSE={rmse_val:.1f}", fontweight='bold')
        ax.set_xlabel('Actual Yield (hg/ha)')
        ax.set_ylabel('Predicted Yield (hg/ha)')
        ax.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'actual_vs_predicted_top3.png'))
    plt.close(fig)
    print("    ✓ Saved actual_vs_predicted_top3.png")

    # Residuals density for the three hybrid options
    residuals_compare = {
        'A Stacking Hybrid':    y_test_raw - unscaled_preds['Option A: Stacking Hybrid'],
        'B BiLSTM-XGB Hybrid':  y_test_raw - unscaled_preds['Option B: BiLSTM-XGBoost Hybrid'],
        'C UniLSTM-XGB Hybrid': y_test_raw - unscaled_preds['Option C: UniLSTM-XGBoost Hybrid']
    }

    fig, ax = plt.subplots(figsize=(12, 7))
    for name, res in residuals_compare.items():
        sns.kdeplot(res, label=name, fill=True, alpha=0.2, ax=ax)

    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_title('Residual Error Distribution Comparison (Hybrids)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Residual Error (Actual - Predicted, hg/ha)')
    ax.set_ylabel('Density')
    ax.legend()
    fig.savefig(os.path.join(PLOT_DIR, 'residuals_density.png'))
    plt.close(fig)
    print("    ✓ Saved residuals_density.png")

    # Final RMSE and R² bar chart
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    results_sorted_rmse = results_df.sort_values(by='RMSE', ascending=False)
    axes[0].barh(results_sorted_rmse['Model'], results_sorted_rmse['RMSE'],
                 color='skyblue', edgecolor='black')
    axes[0].set_title('Model RMSE (Lower is Better)', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('RMSE (hg/ha)')

    results_sorted_r2 = results_df.sort_values(by='R²', ascending=True)
    axes[1].barh(results_sorted_r2['Model'], results_sorted_r2['R²'],
                 color='lightgreen', edgecolor='black')
    axes[1].set_title('Model R² Score (Higher is Better)', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('R² Score')

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'model_metrics_comparison.png'))
    plt.close(fig)
    print("    ✓ Saved model_metrics_comparison.png")

    print("\n" + "="*80)
    print("  EVALUATION STAGE COMPLETE")
    print(f"  All results saved in: {EVAL_DIR}")
    print("="*80)


if __name__ == '__main__':
    run_evaluation()
