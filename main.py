"""
=============================================================================
CROP YIELD PREDICTION — MAIN PIPELINE ORCHESTRATOR
=============================================================================
Executes the complete CYP pipeline end-to-end:

  Stage 1: Data Analysis (EDA)
  Stage 2: Data Preprocessing (Clean → Encode → Scale → Split)
  Stage 3: Feature Selection (LASSO + SHO)
  Stage 4: Model Training (1D CNN-Recursive BiLSTM Hybrid)
  Stage 5: Evaluation (RMSE, MSE, MAE, MBE, R², Accuracy, F1, Precision, Recall)

Architecture: Crop Yield Data → Preprocessing → Feature Selection → 
              1D CNN-Recursive BiLSTM → Prediction

Usage:
  python main.py              # Run complete pipeline
  python main.py --stage 3    # Run from stage 3 onwards
  python main.py --only 1     # Run only stage 1
=============================================================================
"""

import os
import sys
import time
import argparse
import warnings
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def banner():
    print("\n")
    print("+" + "="*70 + "+")
    print("|" + " "*10 + "CROP YIELD PREDICTION (CYP) PIPELINE" + " "*23 + " |")
    print("|" + " "*10 + "1D CNN-Recursive BiLSTM Hybrid Model" + " "*23 + " |")
    print("+" + "="*70 + "+")
    print("|  Stage 1: Deep Data Analysis & EDA                                  |")
    print("|  Stage 2: Data Preprocessing (Clean/Encode/Scale)                   |")
    print("|  Stage 3: Feature Selection (LASSO + SHO)                           |")
    print("|  Stage 4: Model Training (CNN-BiLSTM Hybrid)                        |")
    print("|  Stage 5: Model Evaluation (All Metrics)                            |")
    print("+" + "="*70 + "+")
    print()


def run_stage(stage_num, stage_name, func, *args, **kwargs):
    """Execute a pipeline stage with timing and error handling."""
    print(f"\n{'━'*80}")
    print(f"  ▶ STAGE {stage_num}: {stage_name}")
    print(f"{'━'*80}")
    
    start = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"\n  ✅ Stage {stage_num} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ❌ Stage {stage_num} FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='CYP Pipeline Orchestrator')
    parser.add_argument('--stage', type=int, default=1,
                        help='Start from this stage (default: 1)')
    parser.add_argument('--only', type=int, default=None,
                        help='Run only this specific stage')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--sho-iterations', type=int, default=50,
                        help='SHO iterations (default: 50)')
    parser.add_argument('--sho-herd', type=int, default=30,
                        help='SHO herd size (default: 30)')
    args = parser.parse_args()
    
    banner()
    
    total_start = time.time()
    
    # Determine which stages to run
    if args.only:
        stages_to_run = [args.only]
    else:
        stages_to_run = list(range(args.stage, 6))
    
    print(f"  Stages to execute: {stages_to_run}")
    print(f"  Working directory: {BASE_DIR}")
    
    # ── Stage 1: Data Analysis ──
    if 1 in stages_to_run:
        from importlib import import_module
        mod = import_module('01_data_analysis')
        run_stage(1, "DEEP DATA ANALYSIS & EDA", mod.run_analysis)
    
    # ── Stage 2: Preprocessing ──
    if 2 in stages_to_run:
        from importlib import import_module
        mod = import_module('02_data_preprocessing')
        run_stage(2, "DATA PREPROCESSING",
                  mod.run_preprocessing,
                  missing_strategy='smart',
                  outlier_method='clip',
                  outlier_factor=3.0,
                  scaler_type='standard')
    
    # ── Stage 3: Feature Selection ──
    if 3 in stages_to_run:
        from importlib import import_module
        mod = import_module('03_feature_selection')
        run_stage(3, "FEATURE SELECTION (LASSO + SHO)",
                  mod.run_feature_selection,
                  n_herd=args.sho_herd,
                  n_iterations=args.sho_iterations)
    
    # ── Stage 4: Model Training ──
    if 4 in stages_to_run:
        from importlib import import_module
        mod = import_module('04_model_training')
        run_stage(4, "MODEL TRAINING (1D CNN-Recursive BiLSTM)",
                  mod.run_training,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr)
    
    # ── Stage 5: Evaluation ──
    if 5 in stages_to_run:
        from importlib import import_module
        mod = import_module('05_evaluation')
        run_stage(5, "MODEL EVALUATION", mod.run_evaluation)
    
    # ── Summary ──
    total_elapsed = time.time() - total_start
    
    print(f"\n{'━'*80}")
    print(f"  PIPELINE COMPLETE — Total time: {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f} min)")
    print(f"{'━'*80}")
    print(f"\n  📂 Output directories:")
    print(f"     Analysis:     {os.path.join(BASE_DIR, 'analysis_output')}")
    print(f"     Preprocessed: {os.path.join(BASE_DIR, 'preprocessed_data')}")
    print(f"     Features:     {os.path.join(BASE_DIR, 'feature_selection_output')}")
    print(f"     Models:       {os.path.join(BASE_DIR, 'models')}")
    print(f"     Evaluation:   {os.path.join(BASE_DIR, 'evaluation_output')}")
    print()


if __name__ == '__main__':
    main()
