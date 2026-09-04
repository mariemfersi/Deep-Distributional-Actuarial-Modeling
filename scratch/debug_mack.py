import pandas as pd
import numpy as np
import chainladder as cl
from pathlib import Path
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.reserving.data import build_reserving_dataset
from src.reserving.models import fit_mack_for_company

df, obs, fut = build_reserving_dataset()

# Test company 1767
grcode = 1767
print(f"Testing Mack for GRCODE {grcode}")

model = fit_mack_for_company(obs, grcode)
print("ibnr_ shape:", model.ibnr_.shape)
print("ibnr_ values:")
print(model.ibnr_)

print("\nmack_std_err_ values:")
print(model.mack_std_err_)

# Let's inspect ibnr_ dataframe
ibnr_df = model.ibnr_.to_frame()
print("\nibnr_df:")
print(ibnr_df)
print("ibnr_df columns:", ibnr_df.columns)
print("ibnr_df index:", ibnr_df.index)

std_df = model.mack_std_err_.to_frame()
print("\nstd_df:")
print(std_df)
print("std_df columns:", std_df.columns)
print("std_df index:", std_df.index)
