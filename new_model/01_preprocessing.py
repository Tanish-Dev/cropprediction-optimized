"""
=============================================================================
Stage 1: Preprocessing Pipeline for New Dataset (ICRISAT)
=============================================================================
- Melts wide district-level columns into a long format.
- Filters out inactive records (Area <= 0 or Yield <= 0).
- Sorts data by State, District, Crop, Year and constructs 5-year lags.
- Drops incomplete lag rows (~12% of data), leaving 200,001 rows.
- Splitting: Temporal split (Train: 1966-2010, Val: 2011-2013, Test: 2014-2017).
- SafeLabelEncoder: Handles OOV districts/crops in val/test splits.
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


def load_and_melt_dataset(file_path):
    """
    Load ICRISAT wide dataset and melt it into long format.
    """
    print(f"  Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"    Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    id_vars = ['Dist Code', 'Year', 'State Code', 'State Name', 'Dist Name']
    
    # We want to identify all crops that have Area, Production, and Yield columns
    # We will ignore other general land use columns (Fruits, Vegetables etc.) that do not have yield
    crops = ['BARLEY', 'CASTOR', 'CHICKPEA', 'COTTON', 'FINGER MILLET', 'GROUNDNUT', 
             'KHARIF SORGHUM', 'LINSEED', 'MAIZE', 'MINOR PULSES', 'OILSEEDS', 
             'PEARL MILLET', 'PIGEONPEA', 'RABI SORGHUM', 'RAPESEED AND MUSTARD', 
             'RICE', 'SAFFLOWER', 'SESAMUM', 'SORGHUM', 'SOYABEAN', 'SUGARCANE', 
             'SUNFLOWER', 'WHEAT']
    
    melted_frames = []
    
    for crop in crops:
        area_col = f"{crop} AREA (1000 ha)"
        prod_col = f"{crop} PRODUCTION (1000 tons)"
        yield_col = f"{crop} YIELD (Kg per ha)"
        
        if area_col in df.columns and prod_col in df.columns and yield_col in df.columns:
            sub_df = df[id_vars + [area_col, prod_col, yield_col]].copy()
            sub_df.columns = id_vars + ['Area', 'Production', 'Yield']
            sub_df['Crop'] = crop
            melted_frames.append(sub_df)
            
    melted_df = pd.concat(melted_frames, ignore_index=True)
    print(f"    Melted shape: {melted_df.shape[0]:,} rows × {melted_df.shape[1]} columns")
    return melted_df


def run_preprocessing():
    print("="*80)
    print("  CROP YIELD PREDICTION — DATA PREPROCESSING")
    print("  Stage 1: Clean, Encode, Scale & Temporal Split")
    print("="*80)

    raw_path = os.path.join(BASE_DIR, 'ICRISAT-District Level Data.csv')
    df_melted = load_and_melt_dataset(raw_path)

    # 1. Cleaning and Filtering active records
    print("\n[1/5] Cleaning and filtering active records...")
    df_clean = df_melted[(df_melted['Yield'] > 0) & (df_melted['Area'] > 0)].copy()
    print(f"    Active records (Yield > 0 and Area > 0): {df_clean.shape[0]:,}")

    # Remove duplicates if any
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"    Removed {initial_rows - len(df_clean):,} duplicate rows")

    # 2. Generating lag features (5-year lookback)
    print("\n[2/5] Engineering lag features (5-year lookback)...")
    # Sort chronologically within each group
    df_clean = df_clean.sort_values(by=['State Name', 'Dist Name', 'Crop', 'Year']).reset_index(drop=True)
    
    lookback = 5
    for i in range(1, lookback + 1):
        df_clean[f'yield_lag_{i}'] = df_clean.groupby(['State Name', 'Dist Name', 'Crop'])['Yield'].shift(i)
        df_clean[f'area_lag_{i}'] = df_clean.groupby(['State Name', 'Dist Name', 'Crop'])['Area'].shift(i)

    # Drop rows with incomplete lags
    df_lagged = df_clean.dropna(subset=[f'yield_lag_{i}' for i in range(1, lookback + 1)] + 
                                       [f'area_lag_{i}' for i in range(1, lookback + 1)]).copy()
    print(f"    Dataset shape after dropping incomplete lags: {df_lagged.shape[0]:,}")

    # Save CSV version of preprocessed dataset (useful for dashboard/analysis)
    df_lagged.to_csv(os.path.join(OUTPUT_DIR, 'preprocessed_data.csv'), index=False)
    print("    Saved preprocessed_data.csv")

    # 3. Temporal Split
    print("\n[3/5] Performing chronological/temporal split...")
    # Train: 1966 - 2010
    # Val:   2011 - 2013
    # Test:  2014 - 2017
    train_df = df_lagged[df_lagged['Year'] <= 2010].copy()
    val_df   = df_lagged[(df_lagged['Year'] >= 2011) & (df_lagged['Year'] <= 2013)].copy()
    test_df  = df_lagged[df_lagged['Year'] >= 2014].copy()

    print(f"    Train split:      {len(train_df):,} samples ({df_lagged['Year'].min()} - 2010)")
    print(f"    Validation split: {len(val_df):,} samples (2011 - 2013)")
    print(f"    Test split:       {len(test_df):,} samples (2014 - {df_lagged['Year'].max()})")

    # 4. Safe Encoding of Categoricals (Train only fit, transform all)
    print("\n[4/5] Encoding categorical variables safely (OOV handled)...")
    encoders = {}
    categorical_cols = ['State Name', 'Dist Name', 'Crop']
    
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
    numeric_features = ['Year', 'Area'] + [f'yield_lag_{i}' for i in range(1, lookback + 1)] + \
                                          [f'area_lag_{i}' for i in range(1, lookback + 1)]
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit on training data
    train_features_scaled = feature_scaler.fit_transform(train_df[numeric_features])
    train_target_scaled = target_scaler.fit_transform(train_df[['Yield']]).flatten()

    # Transform validation and test data
    val_features_scaled = feature_scaler.transform(val_df[numeric_features])
    val_target_scaled = target_scaler.transform(val_df[['Yield']]).flatten()

    test_features_scaled = feature_scaler.transform(test_df[numeric_features])
    test_target_scaled = target_scaler.transform(test_df[['Yield']]).flatten()

    # Construct final training arrays
    # Columns sequence in X arrays: [Year, Area, yield_lags (1..5), area_lags (1..5), State_enc, Dist_enc, Crop_enc]
    X_train_num = train_features_scaled
    X_train_cat = train_df[[f'{c}_encoded' for c in categorical_cols]].values
    X_train = np.hstack([X_train_num, X_train_cat])
    y_train = train_target_scaled
    y_train_raw = train_df['Yield'].values

    X_val_num = val_features_scaled
    X_val_cat = val_df[[f'{c}_encoded' for c in categorical_cols]].values
    X_val = np.hstack([X_val_num, X_val_cat])
    y_val = val_target_scaled
    y_val_raw = val_df['Yield'].values

    X_test_num = test_features_scaled
    X_test_cat = test_df[[f'{c}_encoded' for c in categorical_cols]].values
    X_test = np.hstack([X_test_num, X_test_cat])
    y_test = test_target_scaled
    y_test_raw = test_df['Yield'].values

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
