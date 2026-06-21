"""
=============================================================================
Module 03: Feature Selection — LASSO with Selfish Herd Optimisation (SHO)
=============================================================================
Implements the architecture's Feature Selection stage:
  1. LASSO Regression for initial feature importance ranking
  2. Selfish Herd Optimisation (SHO) meta-heuristic to fine-tune
     feature subset selection for optimal model performance
  3. Feature importance visualisation and selected feature export
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
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(BASE_DIR, 'preprocessed_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'feature_selection_output')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)


# ── LASSO Feature Selection ──────────────────────────────────────────────

def lasso_feature_selection(X_train, y_train, X_val, y_val, feature_names,
                             n_alphas=100, cv_folds=5):
    """
    Perform LASSO-based feature selection with cross-validation.
    
    Returns:
      - feature_importance: dict mapping feature names to LASSO coefficients
      - selected_features: list of features with non-zero coefficients
      - best_alpha: optimal regularisation parameter
      - lasso_model: fitted Lasso model
    """
    print("\n  [LASSO Feature Selection]")
    
    # Use LassoCV to find optimal alpha
    # sklearn 1.9+: 'alphas' accepts int (number of alphas) or array
    lasso_cv = LassoCV(n_alphas=n_alphas, cv=cv_folds, random_state=42,
                        max_iter=10000)
    lasso_cv.fit(X_train, y_train)
    best_alpha = lasso_cv.alpha_
    
    print(f"    Best alpha (CV): {best_alpha:.6f}")
    
    # Fit final LASSO with best alpha
    lasso = Lasso(alpha=best_alpha, max_iter=10000, random_state=42)
    lasso.fit(X_train, y_train)
    
    # Extract feature importances
    coefficients = lasso.coef_
    feature_importance = {}
    for name, coef in zip(feature_names, coefficients):
        feature_importance[name] = coef
    
    # Sort by absolute importance
    sorted_features = sorted(feature_importance.items(), 
                              key=lambda x: abs(x[1]), reverse=True)
    
    # Selected features (non-zero coefficients)
    selected = [name for name, coef in sorted_features if abs(coef) > 1e-8]
    eliminated = [name for name, coef in sorted_features if abs(coef) <= 1e-8]
    
    print(f"    Total features: {len(feature_names)}")
    print(f"    Selected features: {len(selected)}")
    print(f"    Eliminated features: {len(eliminated)}")
    
    print(f"\n    Feature Rankings (by |coefficient|):")
    for rank, (name, coef) in enumerate(sorted_features, 1):
        status = '✓' if abs(coef) > 1e-8 else '✗'
        print(f"      {rank}. {status} {name}: {coef:.6f}")
    
    # Evaluate on validation set
    y_pred_val = lasso.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_r2 = r2_score(y_val, y_pred_val)
    print(f"\n    LASSO Validation RMSE: {val_rmse:.4f}")
    print(f"    LASSO Validation R²:   {val_r2:.4f}")
    
    return feature_importance, selected, best_alpha, lasso


# ── Selfish Herd Optimisation (SHO) ──────────────────────────────────────

class SelfishHerdOptimiser:
    """
    Selfish Herd Optimisation (SHO) meta-heuristic for feature subset selection.
    
    Inspired by the selfish herd theory: individuals move towards the centre
    of the group to reduce predation risk. Applied to feature selection,
    each "animal" represents a binary feature mask, and the fitness is
    the model's prediction performance on the validation set.
    
    Reference: Fausto et al., "From ants to whales: metaheuristics for
    all tastes" (2019) — SHO variant.
    """
    
    def __init__(self, n_features, n_herd=30, n_iterations=50,
                 n_predators=5, random_state=42):
        self.n_features = n_features
        self.n_herd = n_herd
        self.n_iterations = n_iterations
        self.n_predators = n_predators
        self.rng = np.random.RandomState(random_state)
        
        # Initialise herd (binary masks)
        self.herd = self.rng.randint(0, 2, size=(n_herd, n_features)).astype(float)
        
        # Ensure at least one feature is selected per individual
        for i in range(n_herd):
            if self.herd[i].sum() == 0:
                self.herd[i, self.rng.randint(0, n_features)] = 1
        
        self.fitness = np.full(n_herd, -np.inf)
        self.best_position = None
        self.best_fitness = -np.inf
        self.history = []
    
    def _evaluate_fitness(self, mask, X_train, y_train, X_val, y_val):
        """Evaluate fitness of a feature subset using LASSO regression."""
        selected_idx = np.where(mask > 0.5)[0]
        
        if len(selected_idx) == 0:
            return -np.inf
        
        X_train_sub = X_train[:, selected_idx]
        X_val_sub = X_val[:, selected_idx]
        
        try:
            model = Lasso(alpha=0.01, max_iter=5000, random_state=42)
            model.fit(X_train_sub, y_train)
            y_pred = model.predict(X_val_sub)
            
            # Fitness = R² score (penalised by number of features for parsimony)
            r2 = r2_score(y_val, y_pred)
            n_selected = len(selected_idx)
            parsimony_penalty = 0.01 * n_selected / self.n_features
            fitness = r2 - parsimony_penalty
            
            return fitness
        except Exception:
            return -np.inf
    
    def optimise(self, X_train, y_train, X_val, y_val, feature_names=None):
        """Run the SHO optimisation loop."""
        print(f"\n  [Selfish Herd Optimisation]")
        print(f"    Herd size:     {self.n_herd}")
        print(f"    Iterations:    {self.n_iterations}")
        print(f"    Predators:     {self.n_predators}")
        print(f"    Feature space: {self.n_features}")
        
        # Evaluate initial fitness
        for i in range(self.n_herd):
            self.fitness[i] = self._evaluate_fitness(
                self.herd[i], X_train, y_train, X_val, y_val)
            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.herd[i].copy()
        
        self.history.append(self.best_fitness)
        
        for iteration in range(self.n_iterations):
            # Sort by fitness (best first)
            sorted_idx = np.argsort(-self.fitness)
            
            # Identify centre of herd (weighted mean of top performers)
            top_k = max(3, self.n_herd // 3)
            top_positions = self.herd[sorted_idx[:top_k]]
            herd_centre = top_positions.mean(axis=0)
            
            # ── Movement Phase: Move towards herd centre ──
            for i in range(self.n_herd):
                # Movement intensity based on rank
                rank = np.where(sorted_idx == i)[0][0]
                movement_rate = 0.5 * (rank / self.n_herd)  # Worse → more movement
                
                # Move towards centre (probabilistic)
                for j in range(self.n_features):
                    if self.rng.random() < movement_rate:
                        self.herd[i, j] = 1 if herd_centre[j] > 0.5 else 0
                
                # Ensure at least one feature
                if self.herd[i].sum() == 0:
                    self.herd[i, self.rng.randint(0, self.n_features)] = 1
            
            # ── Predation Phase: Replace worst individuals ──
            worst_idx = sorted_idx[-self.n_predators:]
            for idx in worst_idx:
                # Replace with mutation of best position
                new_pos = self.best_position.copy()
                n_mutations = max(1, int(self.n_features * 0.2 * self.rng.random()))
                mutation_bits = self.rng.choice(self.n_features, n_mutations, replace=False)
                for bit in mutation_bits:
                    new_pos[bit] = 1 - new_pos[bit]
                
                if new_pos.sum() == 0:
                    new_pos[self.rng.randint(0, self.n_features)] = 1
                
                self.herd[idx] = new_pos
            
            # ── Evaluate fitness ──
            for i in range(self.n_herd):
                self.fitness[i] = self._evaluate_fitness(
                    self.herd[i], X_train, y_train, X_val, y_val)
                if self.fitness[i] > self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_position = self.herd[i].copy()
            
            self.history.append(self.best_fitness)
            
            if (iteration + 1) % 10 == 0 or iteration == 0:
                n_selected = int(self.best_position.sum())
                print(f"    Iteration {iteration+1:3d}/{self.n_iterations}: "
                      f"Best fitness = {self.best_fitness:.6f} | "
                      f"Features = {n_selected}")
        
        # Final results
        selected_mask = self.best_position > 0.5
        selected_idx = np.where(selected_mask)[0]
        
        if feature_names is not None:
            selected_names = [feature_names[i] for i in selected_idx]
        else:
            selected_names = [f"feature_{i}" for i in selected_idx]
        
        print(f"\n    SHO Optimisation Complete!")
        print(f"    Best fitness (R² - parsimony): {self.best_fitness:.6f}")
        print(f"    Selected features ({len(selected_names)}):")
        for name in selected_names:
            print(f"      ✓ {name}")
        
        return selected_idx, selected_names, self.best_position


# ── Visualisation ─────────────────────────────────────────────────────────

def plot_lasso_importance(feature_importance, save_dir):
    """Plot LASSO feature importance bar chart."""
    sorted_feats = sorted(feature_importance.items(), 
                           key=lambda x: abs(x[1]), reverse=True)
    names = [f[0] for f in sorted_feats]
    values = [f[1] for f in sorted_feats]
    abs_values = [abs(v) for v in values]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute importance
    axes[0].barh(names[::-1], abs_values[::-1], color='steelblue', edgecolor='white')
    axes[0].set_title('LASSO — Feature Importance (|Coefficient|)', fontweight='bold')
    axes[0].set_xlabel('Absolute Coefficient')
    
    # Signed coefficients
    axes[1].barh(names[::-1], values[::-1], color=colors[::-1], edgecolor='white')
    axes[1].axvline(x=0, color='black', linewidth=0.8)
    axes[1].set_title('LASSO — Feature Coefficients (Signed)', fontweight='bold')
    axes[1].set_xlabel('Coefficient Value')
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'lasso_feature_importance.png'))
    plt.close(fig)
    print(f"    ✓ Saved: lasso_feature_importance.png")


def plot_sho_convergence(history, save_dir):
    """Plot SHO convergence curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(history)), history, marker='o', markersize=3,
            linewidth=2, color='#2980b9')
    ax.fill_between(range(len(history)), history, alpha=0.1, color='#2980b9')
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Best Fitness (R² - Parsimony)', fontweight='bold')
    ax.set_title('SHO Convergence Curve', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'sho_convergence.png'))
    plt.close(fig)
    print(f"    ✓ Saved: sho_convergence.png")


def plot_feature_comparison(lasso_selected, sho_selected, all_features, save_dir):
    """Compare LASSO vs SHO feature selection."""
    lasso_set = set(lasso_selected)
    sho_set = set(sho_selected)
    all_set = set(all_features)
    
    data = []
    for feat in all_features:
        data.append({
            'Feature': feat,
            'LASSO': 'Yes' if feat in lasso_set else '-',
            'SHO':   'Yes' if feat in sho_set else '-',
            'Both':  'Yes' if feat in (lasso_set & sho_set) else '-',
        })
    
    comparison_df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(8, max(3, len(all_features)*0.4)))
    ax.axis('off')
    table = ax.table(cellText=comparison_df.values,
                     colLabels=comparison_df.columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Colour code header
    for j in range(len(comparison_df.columns)):
        table[0, j].set_facecolor('#3498db')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Feature Selection Comparison: LASSO vs SHO', 
                 fontweight='bold', fontsize=14, pad=30)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'feature_selection_comparison.png'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Saved: feature_selection_comparison.png")


# ── Main Execution ────────────────────────────────────────────────────────

def run_feature_selection(n_herd=30, n_iterations=50, n_predators=5):
    """Execute the complete feature selection pipeline."""
    print("="*80)
    print("  CROP YIELD PREDICTION — FEATURE SELECTION")
    print("  Module 03: LASSO + SHO Meta-Heuristic")
    print("="*80)
    
    # 1. Load preprocessed data
    print("\n[1/5] Loading preprocessed data...")
    splits = np.load(os.path.join(PREPROCESSED_DIR, 'train_test_splits.npz'))
    X_train = splits['X_train']
    y_train = splits['y_train']
    X_val   = splits['X_val']
    y_val   = splits['y_val']
    X_test  = splits['X_test']
    y_test  = splits['y_test']
    
    with open(os.path.join(PREPROCESSED_DIR, 'feature_columns.pkl'), 'rb') as f:
        col_info = pickle.load(f)
    feature_names = col_info['feature_cols']
    
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Validation samples: {X_val.shape[0]:,}")
    print(f"  Features: {len(feature_names)} → {feature_names}")
    
    # 2. LASSO Feature Selection
    print("\n[2/5] Running LASSO feature selection...")
    feature_importance, lasso_selected, best_alpha, lasso_model = \
        lasso_feature_selection(X_train, y_train, X_val, y_val, feature_names)
    
    # 3. SHO Optimisation
    print("\n[3/5] Running Selfish Herd Optimisation...")
    sho = SelfishHerdOptimiser(
        n_features=X_train.shape[1],
        n_herd=n_herd,
        n_iterations=n_iterations,
        n_predators=n_predators,
    )
    sho_selected_idx, sho_selected_names, best_mask = \
        sho.optimise(X_train, y_train, X_val, y_val, feature_names)
    
    # 4. Generate plots
    print("\n[4/5] Generating visualisations...")
    plot_lasso_importance(feature_importance, PLOT_DIR)
    plot_sho_convergence(sho.history, PLOT_DIR)
    plot_feature_comparison(lasso_selected, sho_selected_names, feature_names, PLOT_DIR)
    
    # 5. Save results
    print("\n[5/5] Saving feature selection results...")
    
    # Use SHO-selected features as the final selection
    final_selected_idx = sho_selected_idx
    final_selected_names = sho_selected_names
    
    # If SHO selected fewer meaningful features, fall back to LASSO
    if len(final_selected_names) < 2:
        print("    ⚠ SHO selected too few features, falling back to LASSO selection")
        final_selected_names = lasso_selected
        final_selected_idx = np.array([feature_names.index(f) for f in lasso_selected])
    
    # Create filtered datasets
    X_train_selected = X_train[:, final_selected_idx]
    X_val_selected = X_val[:, final_selected_idx]
    X_test_selected = X_test[:, final_selected_idx]
    
    results = {
        'selected_feature_names': final_selected_names,
        'selected_feature_indices': final_selected_idx,
        'lasso_importance': feature_importance,
        'lasso_selected': lasso_selected,
        'sho_selected': sho_selected_names,
        'sho_best_mask': best_mask,
        'sho_history': sho.history,
        'best_alpha': best_alpha,
    }
    
    with open(os.path.join(OUTPUT_DIR, 'feature_selection_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    np.savez(os.path.join(OUTPUT_DIR, 'selected_features_data.npz'),
             X_train=X_train_selected, y_train=y_train,
             X_val=X_val_selected, y_val=y_val,
             X_test=X_test_selected, y_test=y_test)
    
    print(f"\n    ✓ feature_selection_results.pkl")
    print(f"    ✓ selected_features_data.npz")
    
    print("\n" + "="*80)
    print("  FEATURE SELECTION COMPLETE")
    print(f"  Final selected features ({len(final_selected_names)}):")
    for name in final_selected_names:
        print(f"    ✓ {name}")
    print(f"  Artifacts saved to: {OUTPUT_DIR}")
    print("="*80)
    
    return results


if __name__ == '__main__':
    run_feature_selection()
