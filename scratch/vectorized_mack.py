import pandas as pd
import numpy as np
import chainladder as cl
from pathlib import Path
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.reserving.data import build_reserving_dataset

df, obs, fut = build_reserving_dataset()

print("=== VECTORIZED MACK FIT ===")

# Build full observed triangle with GRCODE index
tri = cl.Triangle(
    data=obs,
    origin="AccidentYear",
    development="DevelopmentYear",
    index=["GRCODE"],
    columns=["CumPaidLoss"],
    cumulative=True
)

model = cl.MackChainladder()
model.fit(tri)

print("Fit finished!")

# Get IBNR and std err
ibnr_df = model.ibnr_.to_frame()
std_df = model.mack_std_err_.to_frame()

print("IBNR shape:", ibnr_df.shape)
print("std_df shape:", std_df.shape)

# Let's inspect ground truth
obs_copy = obs.copy()
fut_copy = fut.copy()

obs_copy["AY"] = obs_copy["AccidentYear"].dt.year
fut_copy["AY"] = fut_copy["AccidentYear"].dt.year

true_ult = fut_copy[fut_copy["DevelopmentLag"] == 10].groupby(["GRCODE", "AY"])["CumPaidLoss"].first()
paid_eval = obs_copy.sort_values("DevelopmentLag").groupby(["GRCODE", "AY"])["CumPaidLoss"].last()
true_ibnr = true_ult - paid_eval

print("True IBNR count:", len(true_ibnr.dropna()))

# Let's evaluate coverage for every company and accident year
results = []
z_90 = 1.645

# Reset index on ibnr_df and std_df
# ibnr_df index is (GRCODE, AccidentYear)
# std_df index is (GRCODE, AccidentYear)
ibnr_series = ibnr_df.iloc[:, 0]
if 9999 in std_df.columns:
    std_series = std_df[9999]
else:
    std_series = std_df.iloc[:, -1]

# Build evaluation dataframe
eval_df = pd.DataFrame({
    "ibnr_mack": ibnr_series,
    "std_err": std_series
})

# Flatten MultiIndex to (grcode, ay_int)
grcodes_idx = [idx[0] for idx in eval_df.index]
ay_idx = [int(str(idx[1])[:4]) for idx in eval_df.index]

eval_df["GRCODE"] = grcodes_idx
eval_df["AY"] = ay_idx
eval_df = eval_df.set_index(["GRCODE", "AY"])

eval_df["ibnr_reel"] = true_ibnr

eval_df = eval_df.dropna()

print(f"Total valid evaluable observations: {len(eval_df)}")

# Calculate metrics
eval_df["lower_90"] = eval_df["ibnr_mack"] - z_90 * eval_df["std_err"]
eval_df["upper_90"] = eval_df["ibnr_mack"] + z_90 * eval_df["std_err"]
eval_df["covered_90"] = (eval_df["ibnr_reel"] >= eval_df["lower_90"]) & (eval_df["ibnr_reel"] <= eval_df["upper_90"])

eval_df["ratio"] = eval_df["ibnr_mack"] / eval_df["ibnr_reel"]
eval_df["under_covered"] = eval_df["ibnr_reel"] > eval_df["upper_90"]
eval_df["over_covered"] = eval_df["ibnr_reel"] < eval_df["lower_90"]

print("\n--- PORTFOLIO-WIDE RESULTS ---")
print(f"Number of companies: {eval_df.index.get_level_values('GRCODE').nunique()}")
print(f"Number of observations: {len(eval_df)}")
print(f"Median ratio Mack / Reel: {eval_df['ratio'].median():.3f}")
print(f"Empirical Coverage Rate (90%): {eval_df['covered_90'].mean():.1%}")
print(f"Under-coverage (real > upper): {eval_df['under_covered'].mean():.1%}")
print(f"Over-coverage (real < lower): {eval_df['over_covered'].mean():.1%}")
print(f"Mack avg width: ${eval_df['upper_90'].mean() - eval_df['lower_90'].mean():,.2f}")

print("\n--- STATE FARM (GRCODE 1767) ---")
sf_df = eval_df.loc[1767]
print(sf_df[["ibnr_mack", "ibnr_reel", "ratio", "covered_90"]])
print(f"State Farm Coverage: {sf_df['covered_90'].mean():.1%} ({sf_df['covered_90'].sum()}/{len(sf_df)})")

print("\n--- CONFORMAL PREDICTION ---")
from src.reserving.models import split_conformal_calibration
test_df, q_hat = split_conformal_calibration(eval_df.reset_index(), alpha=0.10, calib_frac=0.5, seed=123)
print(f"Calibrated q_hat factor: {q_hat:.2f}")
print(f"Test set size: {len(test_df)}")
print(f"Conformal coverage on test set: {test_df['covered_conformal'].mean():.1%}")
mack_w = (test_df["upper_90"] - test_df["lower_90"]).mean()
conf_w = (test_df["upper_conformal"] - test_df["lower_conformal"]).mean()
print(f"Mack avg width on test set: ${mack_w:,.2f}")
print(f"Conformal avg width on test set: ${conf_w:,.2f}")
