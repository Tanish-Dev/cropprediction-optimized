"""
=============================================================================
Stage 1: Preprocessing Pipeline for Old Dataset in new_model2
=============================================================================
- Sorts data by Area (Country), Item (Crop), Year and constructs 5-year lags
  for Yield, Rainfall, Pesticides, and Temperature features.
- Drops incomplete lag rows to guarantee contiguous sequences.
- Splitting: Temporal split (Train: 1990-2007, Val: 2008-2010, Test: 2011-2013).
- SafeLabelEncoder: Handles OOV countries/crops in val/test splits.
- Leakage-free Scaling: Fit scalers on Train only, transform all splits.
=============================================================================
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'preprocessed_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)


class SafeLabelEncoder:
    """
    Custom Label Encoder that handles unseen categories (OOV) gracefully
    by mapping them to a reserved <UNKNOWN> token.
    """
    def __init__(self, unseen_value='<UNKNOWN>'):
        self.unseen_value = unseen_value
        self.classes_ = None
        self.mapping = {}

    def fit(self, y):
        unique_y = pd.Series(y).dropna().unique()
        self.classes_ = np.append(unique_y, self.unseen_value)
        self.mapping = {val: idx for idx, val in enumerate(self.classes_)}
        return self

    def transform(self, y):
        unknown_idx = self.mapping[self.unseen_value]
        return np.array([self.mapping.get(val, unknown_idx) for val in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        inv_mapping = {idx: val for val, idx in self.mapping.items()}
        return np.array([inv_mapping.get(idx, self.unseen_value) for idx in y])


def run_preprocessing():
    print("="*80)
    print("  CROP YIELD PREDICTION — DATA PREPROCESSING (OLD DATASET)")
    print("  Stage 1: Clean, Encode, Scale & Temporal Split")
    print("="*80)

    raw_path = os.path.join(BASE_DIR, 'yield_df.csv')
    print(f"  Loading dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # Drop index/unnamed columns if present
    unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    print(f"    Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # 1. Cleaning and Filtering active records
    print("\n[1/5] Cleaning and filtering active records...")
    df_clean = df[(df['hg/ha_yield'] > 0)].copy()
    print(f"    Active records (Yield > 0): {df_clean.shape[0]:,}")

    # Remove duplicates if any
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"    Removed {initial_rows - len(df_clean):,} duplicate rows")

    # 2. Generating lag features (5-year lookback)
    print("\n[2/5] Engineering lag features (5-year lookback)...")
    # Sort chronologically within each group of Country (Area) and Crop (Item)
    df_clean = df_clean.sort_values(by=['Area', 'Item', 'Year']).reset_index(drop=True)
    
    lookback = 5
    cols_to_lag = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    
    for col in cols_to_lag:
        for i in range(1, lookback + 1):
            df_clean[f'{col}_lag_{i}'] = df_clean.groupby(['Area', 'Item'])[col].shift(i)

    # Drop rows with incomplete lags
    all_lags = [f'{col}_lag_{i}' for col in cols_to_lag for i in range(1, lookback + 1)]
    df_lagged = df_clean.dropna(subset=all_lags).copy()
    print(f"    Dataset shape after dropping incomplete lags: {df_lagged.shape[0]:,}")

    # Save CSV version of preprocessed dataset (useful for dashboard/analysis)
    df_lagged.to_csv(os.path.join(OUTPUT_DIR, 'preprocessed_data.csv'), index=False)
    print("    Saved preprocessed_data.csv")

    # 3. Temporal Split
    print("\n[3/5] Performing chronological/temporal split...")
    # Train: 1990 - 2007
    # Val:   2008 - 2010
    # Test:  2011 - 2013
    train_df = df_lagged[df_lagged['Year'] <= 2007].copy()
    val_df   = df_lagged[(df_lagged['Year'] >= 2008) & (df_lagged['Year'] <= 2010)].copy()
    test_df  = df_lagged[df_lagged['Year'] >= 2011].copy()

    print(f"    Train split:      {len(train_df):,} samples ({df_lagged['Year'].min()} - 2007)")
    print(f"    Validation split: {len(val_df):,} samples (2008 - 2010)")
    print(f"    Test split:       {len(test_df):,} samples (2011 - {df_lagged['Year'].max()})")

    # 4. Safe Encoding of Categoricals (Train only fit, transform all)
    print("\n[4/5] Encoding categorical variables safely (OOV handled)...")
    encoders = {}
    categorical_cols = ['Area', 'Item'] # Area = Country, Item = Crop
    
    for col in categorical_cols:
        encoder = SafeLabelEncoder()
        # Fit only on train to prevent leakage
        encoder.fit(train_df[col])
        
        # Transform train, val, and test splits
        train_df[f'{col}_encoded'] = encoder.transform(train_df[col])
        val_df[f'{col}_encoded'] = encoder.transform(val_df[col])
        test_df[f'{col}_encoded'] = encoder.transform(test_df[col])
        
        encoders[col] = encoder
        print(f"    Encoded {col}: {len(encoder.classes_)-1} classes seen in train + '<UNKNOWN>' OOV class")

    # 5. Feature Scaling (Train only fit, transform all)
    print("\n[5/5] Scaling numeric features & target (Train only fit, transform all)...")
    
    # Sequence of numeric features: 4 current features + 20 lag features
    numeric_features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'] + all_lags
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit on training data
    train_features_scaled = feature_scaler.fit_transform(train_df[numeric_features])
    train_target_scaled = target_scaler.fit_transform(train_df[['hg/ha_yield']]).flatten()

    # Transform validation and test data
    val_features_scaled = feature_scaler.transform(val_df[numeric_features])
    val_target_scaled = target_scaler.transform(val_df[['hg/ha_yield']]).flatten()

    test_features_scaled = feature_scaler.transform(test_df[numeric_features])
    test_target_scaled = target_scaler.transform(test_df[['hg/ha_yield']]).flatten()

    # Construct final training arrays
    # Columns sequence in X arrays: [numeric_features, Area_encoded, Item_encoded]
    X_train_num = train_features_scaled
    X_train_cat = train_df[[f'{c}_encoded' for c in categorical_cols]].values
    X_train = np.hstack([X_train_num, X_train_cat])
    y_train = train_target_scaled
    y_train_raw = train_df['hg/ha_yield'].values

    X_val_num = val_features_scaled
    X_val_cat = val_df[[f'{c}_encoded' for c in categorical_cols]].values
    X_val = np.hstack([X_val_num, X_val_cat])
    y_val = val_target_scaled
    y_val_raw = val_df['hg/ha_yield'].values

    X_test_num = test_features_scaled
    X_test_cat = test_df[[f'{c}_encoded' for c in categorical_cols]].values
    X_test = np.hstack([X_test_num, X_test_cat])
    y_test = test_target_scaled
    y_test_raw = test_df['hg/ha_yield'].values

    # Export split npz files
    np.savez(os.path.join(OUTPUT_DIR, 'train_test_splits.npz'),
             X_train=X_train, y_train=y_train, y_train_raw=y_train_raw,
             X_val=X_val, y_val=y_val, y_val_raw=y_val_raw,
             X_test=X_test, y_test=y_test, y_test_raw=y_test_raw)
    
    # Save encoders & scalers
    with open(os.path.join(OUTPUT_DIR, 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
    with open(os.path.join(OUTPUT_DIR, 'scalers.pkl'), 'wb') as f:
        pickle.dump({
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'numeric_features': numeric_features,
            'categorical_features': categorical_cols
        }, f)

    print("\n" + "="*80)
    print("  PREPROCESSING COMPLETE")
    print(f"  Train:      {X_train.shape[0]:,} samples")
    print(f"  Validation: {X_val.shape[0]:,} samples")
    print(f"  Test:       {X_test.shape[0]:,} samples")
    print(f"  Outputs saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    run_preprocessing()
