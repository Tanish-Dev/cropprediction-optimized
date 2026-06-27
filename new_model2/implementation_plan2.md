# Implementation Plan: Crop Yield Prediction for Old Dataset in new_model2

We want to adopt the modern training, modeling, and evaluation flow established in `new_model` (originally for the ICRISAT dataset) and apply it to the old dataset (global crop yield dataset) located in `new_model2`.

## Proposed Changes

### Preprocessing
#### [NEW] [01_preprocessing.py](file:///Users/Tanish/Documents/cropprediction/new_model2/01_preprocessing.py)
This script will preprocess the old dataset `yield_df.csv` in `new_model2`:
- Standardize variables: `Area` (Country, categorical), `Item` (Crop, categorical), `Year` (numeric), `hg/ha_yield` (Yield, target), `average_rain_fall_mm_per_year` (Rainfall), `pesticides_tonnes` (Pesticides), and `avg_temp` (Temperature).
- Sort chronologically by `['Area', 'Item', 'Year']`.
- Construct 5-year lags for the 4 numeric variables (`hg/ha_yield`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, and `avg_temp`), creating 20 lag features.
- Drop incomplete lag rows.
- Split data temporally:
  - **Train**: 1990 - 2007
  - **Validation**: 2008 - 2010
  - **Test**: 2011 - 2013
- Encode categoricals (`Area` and `Item`) using `SafeLabelEncoder` to support unseen values (OOV).
- Apply leakage-free scaling: fit `StandardScaler` on the training split, and transform train, validation, and test splits.
- Export preprocessed splits to `new_model2/preprocessed_data/train_test_splits.npz`.

### Model Training
#### [NEW] [02_train_option_a.py](file:///Users/Tanish/Documents/cropprediction/new_model2/02_train_option_a.py)
Trains three base tabular regressors:
1. **Random Forest Regressor** (Baseline)
2. **XGBoost Regressor** (Gradient Boosting)
3. **MLP Neural Network** with embedding layers for categoricals (`Area` (Country) mapped to a 16-dimensional embedding, `Item` (Crop) mapped to an 8-dimensional embedding).
A **Stacking Meta-Regressor (Ridge)** is trained on validation set predictions to merge base predictions.

#### [NEW] [03_train_lstm_hybrids.py](file:///Users/Tanish/Documents/cropprediction/new_model2/03_train_lstm_hybrids.py)
Constructs sequence inputs of shape `(samples, 5, 4)` and trains:
- **Option B (BiLSTM)**: Bidirectional LSTM branch + Country and Crop Embeddings branch.
- **Option C (UniLSTM)**: Unidirectional LSTM branch + Country and Crop Embeddings branch.
- Extracts spatial-temporal deep features from the pre-trained networks and trains **XGBoost Hybrid Regressors** on top.

### Evaluation and Inversion
#### [NEW] [04_evaluation.py](file:///Users/Tanish/Documents/cropprediction/new_model2/04_evaluation.py)
- Inverse-scales scaled yield predictions back to the original scale (hectograms per hectare, hg/ha).
- Computes metrics: RMSE, MSE, MAE, MBE, R², MAPE, Acc_10%, Acc_20%.
- Generates evaluation reports, sample predictions, and comparative plots (actual vs. predicted, residuals, and metrics comparisons).

### Orchestration and Dashboard
#### [NEW] [main.py](file:///Users/Tanish/Documents/cropprediction/new_model2/main.py)
End-to-end pipeline execution wrapper running stages 1 through 4.

#### [NEW] [dashboard.py](file:///Users/Tanish/Documents/cropprediction/new_model2/dashboard.py)
Interactive Dash dashboard updated to display statistics, metric leaderboards, comparison charts, and samples tailored to the Country and Crop schema of the old dataset.

---

## Verification Plan

### Automated Verification
Run the pipeline with fewer epochs to verify end-to-end pipeline functionality:
```bash
python3 new_model2/main.py --epochs 3
```

Verify that the output files are generated:
- preprocessed data in `new_model2/preprocessed_data`
- model checkpoints in `new_model2/models`
- evaluation reports and plots in `new_model2/evaluation_output`
