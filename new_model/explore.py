import pandas as pd
import numpy as np

path = 'ICRISAT-District Level Data.csv'
df = pd.read_csv(path)

id_vars = ['Dist Code', 'Year', 'State Code', 'State Name', 'Dist Name']
crops = ['BARLEY', 'CASTOR', 'CHICKPEA', 'COTTON', 'FINGER MILLET', 'GROUNDNUT', 
         'KHARIF SORGHUM', 'LINSEED', 'MAIZE', 'MINOR PULSES', 'OILSEEDS', 
         'PEARL MILLET', 'PIGEONPEA', 'RABI SORGHUM', 'RAPESEED AND MUSTARD', 
         'RICE', 'SAFFLOWER', 'SESAMUM', 'SORGHUM', 'SOYABEAN', 'SUGARCANE', 
         'SUNFLOWER', 'WHEAT']

rows = []
for crop in crops:
    area_col = f"{crop} AREA (1000 ha)"
    prod_col = f"{crop} PRODUCTION (1000 tons)"
    yield_col = f"{crop} YIELD (Kg per ha)"
    
    if area_col in df.columns and prod_col in df.columns and yield_col in df.columns:
        sub_df = df[id_vars + [area_col, prod_col, yield_col]].copy()
        sub_df.columns = id_vars + ['Area', 'Production', 'Yield']
        sub_df['Crop'] = crop
        rows.append(sub_df)

clean_df = pd.concat(rows, ignore_index=True)
clean_df = clean_df[(clean_df['Yield'] > 0) & (clean_df['Area'] > 0)].copy()

# Sort by State, District, Crop, and Year
clean_df = clean_df.sort_values(by=['State Name', 'Dist Name', 'Crop', 'Year']).reset_index(drop=True)

# Create lag features for Yield and Area
lookback = 5
for i in range(1, lookback + 1):
    clean_df[f'yield_lag_{i}'] = clean_df.groupby(['State Name', 'Dist Name', 'Crop'])['Yield'].shift(i)
    clean_df[f'area_lag_{i}'] = clean_df.groupby(['State Name', 'Dist Name', 'Crop'])['Area'].shift(i)

# Let's drop rows that have any NaN values in lags
lagged_df = clean_df.dropna(subset=[f'yield_lag_{i}' for i in range(1, lookback + 1)] + 
                                   [f'area_lag_{i}' for i in range(1, lookback + 1)]).copy()

print("Lagged dataset shape:", lagged_df.shape)
print("Columns:", list(lagged_df.columns))
print("Sample row:")
print(lagged_df.iloc[0])
