import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.reserving.data import load_raw_reserving_data, split_observed_future

# Load data
df = load_raw_reserving_data()
observed, future = split_observed_future(df)

# Test with company 1767
company = observed[observed['GRCODE'] == 1767].copy()
print(f"Company shape: {company.shape}")
print(f"Columns: {company.columns.tolist()}")

# Convert datetime to integers
company['origin_year'] = company['AccidentYear'].dt.year if pd.api.types.is_datetime64_any_dtype(company['AccidentYear']) else company['AccidentYear']

# Get unique years
dev_lags = sorted(company['DevelopmentLag'].unique())
origin_years = sorted(company['origin_year'].unique())

print(f"Dev lags: {dev_lags}")
print(f"Origin years: {origin_years}")

# Build triangle
n_origins = len(origin_years)
n_devs = len(dev_lags)
triangle_values = [[0] * n_devs for _ in range(n_origins)]

for i, origin_year in enumerate(origin_years):
    for j, dev_lag in enumerate(dev_lags):
        cell_data = company[
            (company['origin_year'] == origin_year) & 
            (company['DevelopmentLag'] == dev_lag)
        ]
        if len(cell_data) > 0:
            triangle_values[i][j] = cell_data['CumPaidLoss'].iloc[0]

print(f"\nTriangle row 0 (1998): {triangle_values[0]}")
print(f"Triangle row 1 (1999): {triangle_values[1]}")
