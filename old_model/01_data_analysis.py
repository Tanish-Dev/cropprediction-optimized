"""
=============================================================================
Module 01: Deep Data Analysis & Exploratory Data Analysis (EDA)
=============================================================================
Performs comprehensive analysis of all 9 agricultural datasets:
  - Dataset dimensions, data types, memory usage
  - Missing values analysis (counts, percentages, patterns)
  - Duplicate detection
  - Sample distribution analysis
  - Statistical summaries (mean, std, skewness, kurtosis)
  - Correlation analysis
  - Comparison between base and update datasets
  - Generates publication-quality visualisation plots
=============================================================================
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_output')
PLOT_DIR  = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Plot styling
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'figure.dpi': 150,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
})
sns.set_theme(style='whitegrid', palette='muted')


# ── Dataset Registry ──────────────────────────────────────────────────────
DATASETS = {
    'yield':            os.path.join(BASE_DIR, 'yield.csv'),
    'yield_df':         os.path.join(BASE_DIR, 'yield_df.csv'),
    'yield_update':     os.path.join(BASE_DIR, 'yield_update.csv'),
    'rainfall':         os.path.join(BASE_DIR, 'rainfall.csv'),
    'rainfall_update':  os.path.join(BASE_DIR, 'rainfall_update.csv'),
    'temp':             os.path.join(BASE_DIR, 'temp.csv'),
    'temp_update':      os.path.join(BASE_DIR, 'temp_update.csv'),
    'pesticides':       os.path.join(BASE_DIR, 'pesticides.csv'),
    'pesticides_update':os.path.join(BASE_DIR, 'pesticides_update.csv'),
}


def load_all_datasets():
    """Load all datasets with error handling and return as a dict."""
    loaded = {}
    for name, path in DATASETS.items():
        try:
            df = pd.read_csv(path)
            loaded[name] = df
            print(f"  ✓ Loaded '{name}': {df.shape[0]:,} rows × {df.shape[1]} cols "
                  f"({os.path.getsize(path)/1024:.1f} KB)")
        except Exception as e:
            print(f"  ✗ Failed to load '{name}': {e}")
    return loaded


def analyze_dimensions(datasets):
    """Analyze dimensions, memory usage, and data types for each dataset."""
    print("\n" + "="*80)
    print("  DATASET DIMENSIONS & MEMORY ANALYSIS")
    print("="*80)
    
    summary_rows = []
    for name, df in datasets.items():
        mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
        dtypes_str = ', '.join([f"{dt}({c})" for dt, c in df.dtypes.value_counts().items()])
        summary_rows.append({
            'Dataset': name,
            'Rows': f"{df.shape[0]:,}",
            'Columns': df.shape[1],
            'Memory (MB)': f"{mem_mb:.2f}",
            'Data Types': dtypes_str,
        })
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    return summary_df


def analyze_missing_values(datasets):
    """Detailed missing-value analysis for all datasets."""
    print("\n" + "="*80)
    print("  MISSING VALUES ANALYSIS")
    print("="*80)
    
    all_missing = {}
    for name, df in datasets.items():
        missing = df.isnull().sum()
        total = len(df)
        pct = (missing / total * 100).round(2)
        
        missing_info = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': pct.values
        })
        missing_info = missing_info[missing_info['Missing Count'] > 0]
        
        all_missing[name] = missing_info
        
        total_missing = missing.sum()
        total_cells = df.shape[0] * df.shape[1]
        overall_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
        
        print(f"\n  [{name}]")
        print(f"    Total cells: {total_cells:,} | Missing: {total_missing:,} ({overall_pct:.2f}%)")
        if not missing_info.empty:
            for _, row in missing_info.iterrows():
                print(f"      → {row['Column']}: {int(row['Missing Count']):,} ({row['Missing %']}%)")
        else:
            print(f"      → No missing values ✓")
    
    return all_missing


def analyze_duplicates(datasets):
    """Detect and report duplicate rows in each dataset."""
    print("\n" + "="*80)
    print("  DUPLICATE ANALYSIS")
    print("="*80)
    
    results = {}
    for name, df in datasets.items():
        dup_count = df.duplicated().sum()
        dup_pct = (dup_count / len(df) * 100) if len(df) > 0 else 0
        results[name] = {'count': dup_count, 'pct': dup_pct}
        status = '✓' if dup_count == 0 else '⚠'
        print(f"  {status} [{name}]: {dup_count:,} duplicates ({dup_pct:.2f}%)")
    
    return results


def analyze_statistical_summary(datasets, key_datasets=None):
    """Compute detailed statistical summary for numeric columns."""
    if key_datasets is None:
        key_datasets = ['yield_df', 'yield', 'rainfall', 'temp', 'pesticides']
    
    print("\n" + "="*80)
    print("  STATISTICAL SUMMARY (Numerical Features)")
    print("="*80)
    
    for name in key_datasets:
        if name not in datasets:
            continue
        df = datasets[name]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            continue
        
        print(f"\n  [{name}] — Numeric columns: {numeric_cols}")
        desc = df[numeric_cols].describe().T
        desc['skewness'] = df[numeric_cols].skew()
        desc['kurtosis'] = df[numeric_cols].kurtosis()
        desc['median']   = df[numeric_cols].median()
        desc['IQR']      = desc['75%'] - desc['25%']
        print(desc[['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max',
                     'skewness', 'kurtosis', 'IQR']].to_string())


def analyze_sample_distribution(datasets):
    """Analyze distribution of samples across categories (Area, Item, Year)."""
    print("\n" + "="*80)
    print("  SAMPLE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # yield_df is the primary merged dataset
    if 'yield_df' in datasets:
        df = datasets['yield_df']
        print("\n  [yield_df] — Primary merged dataset")
        
        if 'Area' in df.columns:
            area_counts = df['Area'].value_counts()
            print(f"\n    Countries/Areas: {area_counts.nunique()}")
            print(f"    Top 10 areas by sample count:")
            for area, count in area_counts.head(10).items():
                print(f"      {area}: {count:,}")
        
        if 'Item' in df.columns:
            item_counts = df['Item'].value_counts()
            print(f"\n    Crop Types: {item_counts.nunique()}")
            for item, count in item_counts.items():
                print(f"      {item}: {count:,}")
        
        if 'Year' in df.columns:
            print(f"\n    Year Range: {df['Year'].min()} — {df['Year'].max()}")
            print(f"    Unique Years: {df['Year'].nunique()}")


def compare_base_vs_update(datasets):
    """Compare base datasets with their _update counterparts."""
    print("\n" + "="*80)
    print("  BASE vs UPDATE DATASET COMPARISON")
    print("="*80)
    
    pairs = [
        ('yield', 'yield_update'),
        ('rainfall', 'rainfall_update'),
        ('temp', 'temp_update'),
        ('pesticides', 'pesticides_update'),
    ]
    
    for base_name, update_name in pairs:
        if base_name not in datasets or update_name not in datasets:
            continue
        
        base = datasets[base_name]
        update = datasets[update_name]
        
        same_shape = base.shape == update.shape
        same_cols  = list(base.columns) == list(update.columns)
        
        # Check if content is identical
        try:
            is_identical = base.equals(update)
        except Exception:
            is_identical = False
        
        print(f"\n  [{base_name}] vs [{update_name}]:")
        print(f"    Shape match:   {'✓ Yes' if same_shape else '✗ No'} "
              f"({base.shape} vs {update.shape})")
        print(f"    Columns match: {'✓ Yes' if same_cols else '✗ No'}")
        print(f"    Content identical: {'✓ Yes' if is_identical else '✗ No — Update has changes'}")
        
        if not is_identical and same_shape and same_cols:
            # Find rows that differ
            try:
                diff_mask = (base != update).any(axis=1)
                print(f"    Rows differing: {diff_mask.sum():,} / {len(base):,}")
            except Exception:
                print(f"    (Could not compute row-level diff)")


def analyze_correlations(datasets):
    """Correlation analysis for the primary yield_df dataset."""
    print("\n" + "="*80)
    print("  CORRELATION ANALYSIS")
    print("="*80)
    
    if 'yield_df' not in datasets:
        print("  yield_df not available for correlation analysis.")
        return
    
    df = datasets['yield_df']
    numeric_cols = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 
                    'pesticides_tonnes', 'avg_temp']
    
    available_cols = [c for c in numeric_cols if c in df.columns]
    if len(available_cols) < 2:
        print("  Not enough numeric columns for correlation analysis.")
        return
    
    corr_matrix = df[available_cols].corr()
    print(f"\n  Pearson Correlation Matrix:")
    print(corr_matrix.round(4).to_string())
    
    return corr_matrix


# ── Visualisation Functions ───────────────────────────────────────────────



def plot_correlation_heatmap(corr_matrix):
    """Plot a publication-quality correlation heatmap."""
    if corr_matrix is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title('Feature Correlation Matrix — yield_df', fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'correlation_heatmap.png'))
    plt.close(fig)
    print(f"  ✓ Saved: correlation_heatmap.png")


def plot_distribution_histograms(datasets):
    """Plot feature distributions for yield_df."""
    if 'yield_df' not in datasets:
        return
    
    df = datasets['yield_df']
    numeric_cols = ['hg/ha_yield', 'average_rain_fall_mm_per_year',
                    'pesticides_tonnes', 'avg_temp']
    available = [c for c in numeric_cols if c in df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(available):
        data = df[col].dropna()
        axes[i].hist(data, bins=50, color=sns.color_palette('muted')[i],
                     edgecolor='white', alpha=0.8)
        axes[i].axvline(data.mean(), color='red', linestyle='--', 
                        linewidth=1.5, label=f'Mean: {data.mean():.2f}')
        axes[i].axvline(data.median(), color='green', linestyle='-.',
                        linewidth=1.5, label=f'Median: {data.median():.2f}')
        axes[i].set_title(f'Distribution of {col}', fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    # Hide unused subplots
    for j in range(len(available), 4):
        axes[j].set_visible(False)
    
    fig.suptitle('Feature Distributions — yield_df', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'feature_distributions.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: feature_distributions.png")


def plot_boxplots(datasets):
    """Plot boxplots for outlier detection."""
    if 'yield_df' not in datasets:
        return
    
    df = datasets['yield_df']
    numeric_cols = ['hg/ha_yield', 'average_rain_fall_mm_per_year',
                    'pesticides_tonnes', 'avg_temp']
    available = [c for c in numeric_cols if c in df.columns]
    
    fig, axes = plt.subplots(1, len(available), figsize=(5*len(available), 8))
    if len(available) == 1:
        axes = [axes]
    
    colors = sns.color_palette('Set2', len(available))
    for i, col in enumerate(available):
        bp = axes[i].boxplot(df[col].dropna(), patch_artist=True,
                             boxprops=dict(facecolor=colors[i], alpha=0.7),
                             medianprops=dict(color='red', linewidth=2),
                             whiskerprops=dict(linewidth=1.5),
                             capprops=dict(linewidth=1.5))
        axes[i].set_title(col, fontweight='bold')
        axes[i].set_ylabel('Value')
        
        # Annotate outlier count
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        axes[i].text(0.5, 0.02, f'Outliers: {outliers:,}', 
                     transform=axes[i].transAxes, ha='center', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle('Boxplots for Outlier Detection — yield_df', 
                 fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'boxplots_outliers.png'))
    plt.close(fig)
    print(f"  ✓ Saved: boxplots_outliers.png")


def plot_yield_by_crop(datasets):
    """Plot average yield by crop type."""
    if 'yield_df' not in datasets:
        return
    
    df = datasets['yield_df']
    if 'Item' not in df.columns or 'hg/ha_yield' not in df.columns:
        return
    
    crop_yield = df.groupby('Item')['hg/ha_yield'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(crop_yield)*0.8)))
    bars = ax.barh(crop_yield.index, crop_yield['mean'], 
                   xerr=crop_yield['std'], color=sns.color_palette('viridis', len(crop_yield)),
                   edgecolor='white', alpha=0.85, capsize=3)
    ax.set_xlabel('Average Yield (hg/ha)', fontweight='bold')
    ax.set_title('Average Crop Yield by Item', fontweight='bold', fontsize=14)
    
    # Annotate sample counts
    for i, (idx, row) in enumerate(crop_yield.iterrows()):
        ax.text(row['mean'] + row['std'] + 500, i, f"n={int(row['count']):,}",
                va='center', fontsize=9, color='gray')
    
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'yield_by_crop.png'))
    plt.close(fig)
    print(f"  ✓ Saved: yield_by_crop.png")


def plot_yield_trends(datasets):
    """Plot yield trends over time for top areas."""
    if 'yield_df' not in datasets:
        return
    
    df = datasets['yield_df']
    if 'Year' not in df.columns or 'hg/ha_yield' not in df.columns:
        return
    
    # Top 10 areas by number of samples
    top_areas = df['Area'].value_counts().head(10).index.tolist() if 'Area' in df.columns else []
    
    if top_areas:
        fig, ax = plt.subplots(figsize=(14, 8))
        for area in top_areas:
            area_data = df[df['Area'] == area].groupby('Year')['hg/ha_yield'].mean()
            ax.plot(area_data.index, area_data.values, marker='o', markersize=3, 
                    linewidth=1.5, label=area, alpha=0.8)
        
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Average Yield (hg/ha)', fontweight='bold')
        ax.set_title('Yield Trends Over Time — Top 10 Areas', fontweight='bold', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, 'yield_trends.png'), bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: yield_trends.png")




def generate_report(datasets, dims, missing, dups):
    """Generate a comprehensive text report."""
    report_path = os.path.join(OUTPUT_DIR, 'data_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("  CROP YIELD PREDICTION — COMPREHENSIVE DATA ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("Generated by: 01_data_analysis.py\n")
        f.write(f"Datasets Analyzed: {len(datasets)}\n")
        f.write(f"Total Samples (yield_df): {len(datasets.get('yield_df', []))}\n\n")
        
        # Dimensions summary
        f.write("─"*60 + "\n")
        f.write("DIMENSIONS SUMMARY\n")
        f.write("─"*60 + "\n")
        f.write(dims.to_string(index=False) + "\n\n")
        
        # Missing values
        f.write("─"*60 + "\n")
        f.write("MISSING VALUES\n")
        f.write("─"*60 + "\n")
        for name, info in missing.items():
            if info.empty:
                f.write(f"  {name}: No missing values\n")
            else:
                f.write(f"  {name}:\n")
                f.write(info.to_string(index=False) + "\n")
        
        # Duplicates
        f.write("\n" + "─"*60 + "\n")
        f.write("DUPLICATES\n")
        f.write("─"*60 + "\n")
        for name, info in dups.items():
            f.write(f"  {name}: {info['count']:,} ({info['pct']:.2f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("  END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n  ✓ Report saved: {report_path}")


# ── Main Execution ────────────────────────────────────────────────────────

def run_analysis():
    """Execute the complete data analysis pipeline."""
    print("="*80)
    print("  CROP YIELD PREDICTION — DEEP DATA ANALYSIS")
    print("  Module 01: Exploratory Data Analysis")
    print("="*80)
    
    # 1. Load datasets
    print("\n[1/9] Loading all datasets...")
    datasets = load_all_datasets()
    
    # 2. Dimensions & types
    print("\n[2/9] Analyzing dimensions & data types...")
    dims = analyze_dimensions(datasets)
    
    # 3. Missing values
    print("\n[3/9] Analyzing missing values...")
    missing = analyze_missing_values(datasets)
    
    # 4. Duplicates
    print("\n[4/9] Detecting duplicates...")
    dups = analyze_duplicates(datasets)
    
    # 5. Statistical summary
    print("\n[5/9] Computing statistical summaries...")
    analyze_statistical_summary(datasets)
    
    # 6. Sample distribution
    print("\n[6/9] Analyzing sample distributions...")
    analyze_sample_distribution(datasets)
    
    # 7. Base vs Update comparison
    print("\n[7/9] Comparing base vs update datasets...")
    compare_base_vs_update(datasets)
    
    # 8. Correlation analysis
    print("\n[8/9] Computing correlations...")
    corr_matrix = analyze_correlations(datasets)
    
    # 9. Generate visualisations
    print("\n[9/9] Generating visualisations...")
    plot_correlation_heatmap(corr_matrix)
    plot_distribution_histograms(datasets)
    plot_boxplots(datasets)
    plot_yield_by_crop(datasets)
    plot_yield_trends(datasets)
    
    # Generate report
    generate_report(datasets, dims, missing, dups)
    
    print("\n" + "="*80)
    print("  ANALYSIS COMPLETE")
    print(f"  Plots saved to: {PLOT_DIR}")
    print(f"  Report saved to: {os.path.join(OUTPUT_DIR, 'data_analysis_report.txt')}")
    print("="*80)
    
    return datasets


if __name__ == '__main__':
    run_analysis()
