import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.reserving.data import load_raw_reserving_data, compute_incremental_paid

df = load_raw_reserving_data()
df = compute_incremental_paid(df)

print("=== CHECKING EarnedPremNet FILTERING ===")
# 1. Check min premium per company across all years
min_prem_by_co = df.groupby("GRCODE")["EarnedPremNet"].min()

for thresh in [0, 1000, 10000, 50000, 100000]:
    valid_cos = min_prem_by_co[min_prem_by_co >= thresh].index
    filtered_df = df[df["GRCODE"].isin(valid_cos)].copy()
    
    # number of sequences (GRCODE, AccidentYear)
    # wait: AccidentYear has 10 values per company (1998..2007)
    # wait: 92 companies * 10 years = 920 sequences? Or how many sequences?
    n_seqs = len(filtered_df.groupby(["GRCODE", "AccidentYear"]))
    
    # ScaledIncr
    scaled = filtered_df["IncrementalPaid"] / filtered_df["EarnedPremNet"].clip(lower=1.0)
    skew = scaled.skew()
    
    print(f"Thresh={thresh:7d} -> {len(valid_cos):3d} companies, {n_seqs:4d} sequences, skew={skew:6.2f}, min={scaled.min():8.2f}, max={scaled.max():6.2f}")
