"""
=============================================================================
CROP YIELD PREDICTION — MAIN PIPELINE ORCHESTRATOR (NEW ICRISAT MODEL)
=============================================================================
Executes the complete modern CYP pipeline end-to-end:

  Stage 1: Preprocessing (Clean → Melt → Lag → Encoding → Leakage-free Scaling)
  Stage 2: Option A Training (Tabular RF, XGB, MLP & Stacking Ensemble)
  Stage 3: Option B & C Training (Bidirectional B, Unidirectional C & XGB Hybrids)
  Stage 4: Model Evaluation (Report generation, inverse target scaling, plots)

Usage:
  python3 main.py                  # Run complete pipeline
  python3 main.py --epochs 3       # Run fast pipeline verify
=============================================================================
"""

import os
import sys
import time
import argparse
import warnings
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def banner():
    print("\n")
    print("+" + "="*70 + "+")
    print("|" + " "*10 + "CROP YIELD PREDICTION (CYP) NEW PIPELINE" + " "*20 + " |")
    print("|" + " "*10 + "Stacking (A) & LSTM-XGBoost Hybrids (B/C)" + " "*20 + " |")
    print("+" + "="*70 + "+")
    print("|  Stage 1: Clean, Melt, Lag & Safe Preprocessing                      |")
    print("|  Stage 2: Train Option A (RF, XGB, MLP & Stacking)                   |")
    print("|  Stage 3: Train Option B (BiLSTM-XGB) & Option C (UniLSTM-XGB)         |")
    print("|  Stage 4: Comprehensive Model Evaluation & Inversion                 |")
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
    parser = argparse.ArgumentParser(description='CYP ICRISAT Pipeline Orchestrator')
    parser.add_argument('--stage', type=int, default=1,
                        help='Start from this stage (default: 1)')
    parser.add_argument('--only', type=int, default=None,
                        help='Run only this specific stage')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Deep Learning training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Deep Learning training batch size (default: 256)')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of estimators for RF/XGB (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    args = parser.parse_args()
    
    banner()
    
    total_start = time.time()
    
    # Determine which stages to run
    if args.only:
        stages_to_run = [args.only]
    else:
        stages_to_run = list(range(args.stage, 5))
    
    print(f"  Stages to execute: {stages_to_run}")
    print(f"  Working directory: {BASE_DIR}")
    
    # ── Stage 1: Data Preprocessing ──
    if 1 in stages_to_run:
        from importlib import import_module
        mod = import_module('01_preprocessing')
        run_stage(1, "SAFE PREPROCESSING & TEMPORAL SPLITTING", mod.run_preprocessing)
        
    # ── Stage 2: Option A Training ──
    if 2 in stages_to_run:
        from importlib import import_module
        mod = import_module('02_train_option_a')
        run_stage(2, "OPTION A TRAINING (RF + XGB + MLP + STACKING)", 
                  mod.run_training_option_a,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  n_estimators=args.n_estimators,
                  learning_rate=args.lr)
        
    # ── Stage 3: Options B & C Training ──
    if 3 in stages_to_run:
        from importlib import import_module
        mod = import_module('03_train_lstm_hybrids')
        run_stage(3, "OPTION B & C LSTM HYBRIDS (BILSTM / UNILSTM + XGBOOST)", 
                  mod.run_training_lstm_hybrids,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  n_estimators=args.n_estimators,
                  learning_rate=args.lr)
        
    # ── Stage 4: Evaluation ──
    if 4 in stages_to_run:
        from importlib import import_module
        mod = import_module('04_evaluation')
        run_stage(4, "COMPREHENSIVE EVALUATION (INVERSION + PLOTS)", mod.run_evaluation)
        
    total_elapsed = time.time() - total_start
    print(f"\n{'━'*80}")
    print(f"  PIPELINE COMPLETE — Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'━'*80}")
    print(f"\n  📂 Output directories:")
    print(f"     Preprocessed: {os.path.join(BASE_DIR, 'preprocessed_data')}")
    print(f"     Models:       {os.path.join(BASE_DIR, 'models')}")
    print(f"     Evaluation:   {os.path.join(BASE_DIR, 'evaluation_output')}")
    print()


if __name__ == '__main__':
    main()
