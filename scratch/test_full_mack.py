import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.reserving.data import build_reserving_dataset
from src.reserving.models import fit_mack_for_company

df, obs, fut = build_reserving_dataset()

def evaluate_mack_coverage_fixed(observed: pd.DataFrame, future: pd.DataFrame, grcode: int, z_90: float = 1.645):
    try:
        model = fit_mack_for_company(observed, grcode)

        ibnr = model.ibnr_.to_frame().iloc[:, 0]
        # Convert index to integer year safely
        if hasattr(ibnr.index, "year"):
            ibnr.index = ibnr.index.year
        else:
            ibnr.index = [int(str(x)[:4]) for x in ibnr.index]

        std_df = model.mack_std_err_.to_frame()
        if 9999 in std_df.columns:
            std_err = std_df[9999]
        else:
            std_err = std_df.iloc[:, -1]

        if hasattr(std_err.index, "year"):
            std_err.index = std_err.index.year
        else:
            std_err = std_err.index = [int(str(x)[:4]) for x in std_err.index]

        # Ground truth
        obs_co = observed[observed["GRCODE"] == grcode].copy()
        fut_co = future[future["GRCODE"] == grcode].copy()

        obs_co["AY_year"] = obs_co["AccidentYear"].dt.year if pd.api.types.is_datetime64_any_dtype(obs_co["AccidentYear"]) else obs_co["AccidentYear"]
        fut_co["AY_year"] = fut_co["AccidentYear"].dt.year if pd.api.types.is_datetime64_any_dtype(fut_co["AccidentYear"]) else fut_co["AccidentYear"]

        true_ultimate = fut_co[fut_co["DevelopmentLag"] == 10].set_index("AY_year")["CumPaidLoss"]
        paid_at_eval = obs_co.sort_values("DevelopmentLag").groupby("AY_year")["CumPaidLoss"].last()
        true_ibnr = true_ultimate - paid_at_eval

        results = pd.DataFrame({
            "ibnr_mack": ibnr,
            "ibnr_reel": true_ibnr,
            "std_err": std_err,
        }).dropna()

        if len(results) == 0:
            return None

        results["lower_90"] = results["ibnr_mack"] - z_90 * results["std_err"]
        results["upper_90"] = results["ibnr_mack"] + z_90 * results["std_err"]
        results["covered_90"] = (results["ibnr_reel"] >= results["lower_90"]) & (results["ibnr_reel"] <= results["upper_90"])
        results["grcode"] = grcode

        return results

    except Exception as e:
        return None

print("=== EVALUATING ALL 143 COMPANIES ===")
grcodes = df["GRCODE"].unique()
all_results = []
failures = []
successes = []

for gr in grcodes:
    res = evaluate_mack_coverage_fixed(obs, fut, gr)
    if res is not None and len(res) > 0:
        all_results.append(res)
        successes.append(gr)
    else:
        failures.append(gr)

print(f"Total companies: {len(grcodes)}")
print(f"Successful models: {len(successes)} ({len(successes)/len(grcodes):.1%})")
print(f"Failed models: {len(failures)} ({len(failures)/len(grcodes):.1%})")

full_df = pd.concat(all_results, ignore_index=True)
print(f"Total evaluable observations: {len(full_df)}")

full_df["ratio"] = full_df["ibnr_mack"] / full_df["ibnr_reel"]
# Filter out inf / invalid ratios if any
valid_ratios = full_df["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
full_df["under_covered"] = full_df["ibnr_reel"] > full_df["upper_90"]
full_df["over_covered"] = full_df["ibnr_reel"] < full_df["lower_90"]

print(f"Median ratio pred/reel: {valid_ratios.median():.3f}")
print(f"Global empirical coverage (90%): {full_df['covered_90'].mean():.1%}")
print(f"Under-coverage (real > upper): {full_df['under_covered'].mean():.1%}")
print(f"Over-coverage (real < lower): {full_df['over_covered'].mean():.1%}")
print(f"Average width of Mack interval: ${full_df['upper_90'].mean() - full_df['lower_90'].mean():,.2f}")

print("\n=== CONFORMAL EVALUATION ===")
from src.reserving.models import split_conformal_calibration
test_df, q_hat = split_conformal_calibration(full_df, alpha=0.10, calib_frac=0.5, seed=123)
print(f"Calibrated factor q_hat: {q_hat:.2f}")
print(f"Test set size: {len(test_df)}")
print(f"Conformal coverage on test set: {test_df['covered_conformal'].mean():.1%}")
mack_width_test = (test_df["upper_90"] - test_df["lower_90"]).mean()
conf_width_test = (test_df["upper_conformal"] - test_df["lower_conformal"]).mean()
print(f"Mack avg width on test set: ${mack_width_test:,.2f}")
print(f"Conformal avg width on test set: ${conf_width_test:,.2f}")

print("\n=== STATE FARM SPECIFIC (GRCODE 1767) ===")
sf = evaluate_mack_coverage_fixed(obs, fut, 1767)
if sf is not None:
    sf["ratio"] = sf["ibnr_mack"] / sf["ibnr_reel"]
    print("State Farm Obs count:", len(sf))
    print("State Farm Ratios (by AY):")
    print(sf[["ibnr_mack", "ibnr_reel", "ratio", "covered_90"]])
    print(f"State Farm Coverage: {sf['covered_90'].mean():.1%}")
