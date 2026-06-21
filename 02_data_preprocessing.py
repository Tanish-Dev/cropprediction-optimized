"""
=============================================================================
Module 02: Data Preprocessing Pipeline
=============================================================================
Implements the complete preprocessing pipeline per the CYP architecture:
  1. Load and consolidate datasets (base + update, or yield_df)
  2. Handle missing values (imputation strategies)
  3. Remove duplicates
  4. Label Encoding for categorical variables (Area, Item)
  5. Data Scaling (StandardScaler / MinMaxScaler)
  6. Outlier detection and handling
  7. Export cleaned, scaled dataset ready for feature selection
=============================================================================
"""

import os
import sys
import warnings
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'preprocessed_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_primary_dataset():
    """
    Load the primary merged dataset (yield_df.csv).
    This dataset already has yield, rainfall, pesticides, and temperature merged.
    """
    path = os.path.join(BASE_DIR, 'yield_df.csv')
    df = pd.read_csv(path)
    print(f"  Loaded yield_df: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Columns: {list(df.columns)}")
    return df



def clean_dataset(df):
    """
    Clean the dataset:
      - Drop unnamed/index columns
      - Standardise column names
      - Remove exact duplicate rows
    """
    print("\n  [Cleaning]")
    
    # Drop unnamed index column if exists
    unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"    Dropped {len(unnamed_cols)} unnamed index columns")
    
    # Report initial state
    initial_rows = len(df)
    
    # Remove exact duplicate rows
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"    Removed {removed:,} duplicate rows ({initial_rows:,} → {len(df):,})")
    
    return df


def handle_missing_values(df, strategy='smart'):
    """
    Handle missing values with configurable strategies.
    
    Strategies:
      - 'drop':    Drop rows with any missing values
      - 'mean':    Fill with column mean (numeric only)
      - 'median':  Fill with column median (numeric only)
      - 'smart':   Context-aware imputation (group-based for area/year, then median)
    """
    print(f"\n  [Missing Values — Strategy: '{strategy}']")
    
    missing_before = df.isnull().sum().sum()
    print(f"    Missing cells before: {missing_before:,}")
    
    if missing_before == 0:
        print(f"    No missing values — skipping")
        return df
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if strategy == 'drop':
        df = df.dropna()
    
    elif strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    
    elif strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    
    elif strategy == 'smart':
        # Group-based imputation: fill missing with group mean (by Area)
        for col in numeric_cols:
            if df[col].isnull().sum() > 0 and 'Area' in df.columns:
                # Fill with area-specific mean first
                group_means = df.groupby('Area')[col].transform('mean')
                df[col] = df[col].fillna(group_means)
                
                # Then fill remaining with global median
                df[col] = df[col].fillna(df[col].median())
            elif df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
    
    # Drop any remaining rows with missing values
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"    Remaining missing after imputation: {remaining_missing:,}")
        df = df.dropna()
    
    missing_after = df.isnull().sum().sum()
    print(f"    Missing cells after:  {missing_after:,}")
    print(f"    Final shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    
    return df


def encode_labels(df):
    """
    Apply Label Encoding to categorical columns (Area, Item).
    Returns the encoded DataFrame and the fitted encoders for inverse transform.
    """
    print("\n  [Label Encoding]")
    
    encoders = {}
    categorical_cols = ['Area', 'Item']
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
        
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
        n_classes = len(le.classes_)
        print(f"    {col}: {n_classes} unique classes → encoded to [0, {n_classes-1}]")
    
    return df, encoders


def handle_outliers(df, method='iqr', factor=3.0):
    """
    Handle outliers in numeric features.
    
    Methods:
      - 'iqr':  Remove rows beyond factor*IQR from Q1/Q3
      - 'clip': Clip values to factor*IQR bounds
      - 'none': Skip outlier handling
    """
    print(f"\n  [Outlier Handling — Method: '{method}', Factor: {factor}]")
    
    if method == 'none':
        print(f"    Skipping outlier handling")
        return df
    
    target_cols = ['hg/ha_yield', 'average_rain_fall_mm_per_year',
                   'pesticides_tonnes', 'avg_temp']
    cols_to_process = [c for c in target_cols if c in df.columns]
    
    initial_rows = len(df)
    
    for col in cols_to_process:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        
        if method == 'iqr':
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        elif method == 'clip':
            df[col] = df[col].clip(lower, upper)
        
        print(f"    {col}: {outlier_count:,} outliers "
              f"(bounds: [{lower:.2f}, {upper:.2f}])")
    
    if method == 'iqr':
        print(f"    Rows removed: {initial_rows - len(df):,} "
              f"({initial_rows:,} → {len(df):,})")
    
    return df


def scale_features(df, scaler_type='standard'):
    """
    Apply feature scaling to numeric columns.
    
    Scaler types:
      - 'standard': StandardScaler (zero mean, unit variance)
      - 'minmax':   MinMaxScaler (scale to [0, 1])
    """
    print(f"\n  [Feature Scaling — '{scaler_type}']")
    
    feature_cols = ['hg/ha_yield', 'average_rain_fall_mm_per_year',
                    'pesticides_tonnes', 'avg_temp', 'Year']
    encoded_cols = [c for c in df.columns if c.endswith('_encoded')]
    
    cols_to_scale = [c for c in feature_cols + encoded_cols if c in df.columns]
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Create a copy for scaled values
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    print(f"    Scaled {len(cols_to_scale)} columns: {cols_to_scale}")
    
    return df_scaled, scaler, cols_to_scale


def prepare_train_test_split(df, target_col='hg/ha_yield', test_size=0.2, 
                              val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets.
    """
    print(f"\n  [Train/Validation/Test Split]")
    
    # Define feature columns (exclude original categorical and target)
    exclude_cols = ['Area', 'Item', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction, random_state=random_state)
    
    print(f"    Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"    Target: {target_col}")
    print(f"    Train:      {X_train.shape[0]:,} samples ({(1-test_size)*(1-val_fraction)*100:.0f}%)")
    print(f"    Validation: {X_val.shape[0]:,} samples ({(1-test_size)*val_fraction*100:.0f}%)")
    print(f"    Test:       {X_test.shape[0]:,} samples ({test_size*100:.0f}%)")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_cols': feature_cols,
        'target_col': target_col,
    }


def save_artifacts(df_clean, df_scaled, encoders, scaler, scaled_cols, splits):
    """Save all preprocessing artifacts for downstream modules."""
    print(f"\n  [Saving Artifacts to {OUTPUT_DIR}]")
    
    # Save DataFrames
    df_clean.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_data.csv'), index=False)
    df_scaled.to_csv(os.path.join(OUTPUT_DIR, 'scaled_data.csv'), index=False)
    
    # Save encoders
    with open(os.path.join(OUTPUT_DIR, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save scaler
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump({'scaler': scaler, 'columns': scaled_cols}, f)
    
    # Save splits
    np.savez(os.path.join(OUTPUT_DIR, 'train_test_splits.npz'),
             X_train=splits['X_train'], y_train=splits['y_train'],
             X_val=splits['X_val'], y_val=splits['y_val'],
             X_test=splits['X_test'], y_test=splits['y_test'])
    
    # Save feature column names
    with open(os.path.join(OUTPUT_DIR, 'feature_columns.pkl'), 'wb') as f:
        pickle.dump({
            'feature_cols': splits['feature_cols'],
            'target_col': splits['target_col']
        }, f)
    
    print(f"    ✓ cleaned_data.csv")
    print(f"    ✓ scaled_data.csv")
    print(f"    ✓ label_encoders.pkl")
    print(f"    ✓ scaler.pkl")
    print(f"    ✓ train_test_splits.npz")
    print(f"    ✓ feature_columns.pkl")


# ── Main Execution ────────────────────────────────────────────────────────

def run_preprocessing(missing_strategy='smart', outlier_method='clip', 
                       outlier_factor=3.0, scaler_type='standard',
                       test_size=0.2, val_size=0.1):
    """Execute the complete preprocessing pipeline."""
    print("="*80)
    print("  CROP YIELD PREDICTION — DATA PREPROCESSING")
    print("  Module 02: Clean → Encode → Scale → Split")
    print("="*80)
    
    # 1. Load
    print("\n[1/7] Loading primary dataset...")
    df = load_primary_dataset()
    
    # 2. Clean
    print("\n[2/7] Cleaning dataset...")
    df_clean = clean_dataset(df)
    
    # 3. Handle missing values
    print("\n[3/7] Handling missing values...")
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    
    # 4. Handle outliers
    print("\n[4/7] Handling outliers...")
    df_clean = handle_outliers(df_clean, method=outlier_method, factor=outlier_factor)
    
    # 5. Label encoding
    print("\n[5/7] Encoding categorical variables...")
    df_encoded, encoders = encode_labels(df_clean)
    
    # 6. Scale features
    print("\n[6/7] Scaling features...")
    df_scaled, scaler, scaled_cols = scale_features(df_encoded, scaler_type=scaler_type)
    
    # 7. Train/test split
    print("\n[7/7] Splitting into train/val/test...")
    splits = prepare_train_test_split(
        df_scaled, test_size=test_size, val_size=val_size)
    
    # Save everything
    save_artifacts(df_clean, df_scaled, encoders, scaler, scaled_cols, splits)
    
    print("\n" + "="*80)
    print("  PREPROCESSING COMPLETE")
    print(f"  Clean samples: {len(df_clean):,}")
    print(f"  Features: {len(splits['feature_cols'])}")
    print(f"  Artifacts saved to: {OUTPUT_DIR}")
    print("="*80)
    
    return df_clean, df_scaled, encoders, scaler, splits


if __name__ == '__main__':
    run_preprocessing()
