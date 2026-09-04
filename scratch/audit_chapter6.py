import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.fraud.data import load_fraud_data, prepare_fraud_features, train_test_split_fraud, CATEGORICAL_COLS, NUMERIC_COLS
from src.fraud.models import fit_isolation_forest, evaluate_isolation_forest, fit_supervised_baseline, evaluate_supervised
from src.fraud.graph import build_edge_index, build_edge_index_rare_shared, build_pyg_graph

print("=== 1. DATASET & PREVALENCE CHECK ===")
df = load_fraud_data()
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Missing values: {df.isnull().sum().sum()}")
fraud_prev = df["FraudFound_P"].mean()
print(f"Fraud count: {df['FraudFound_P'].sum()}, Fraud prevalence: {fraud_prev:.4f} ({fraud_prev:.2%})")

print("\n=== 2. DISCRIMINANT SIGNALS ===")
print("AddressChange_Claim fraud rates:")
ac_rates = df.groupby("AddressChange_Claim")["FraudFound_P"].agg(["count", "mean"])
print(ac_rates)

print("\nFault fraud rates:")
fault_rates = df.groupby("Fault")["FraudFound_P"].agg(["count", "mean"])
print(fault_rates)

print("\nDays_Policy_Claim fraud rates:")
dpc_rates = df.groupby("Days_Policy_Claim")["FraudFound_P"].agg(["count", "mean"])
print(dpc_rates)

print("\n=== 3. UNSUPERVISED & SUPERVISED MODELS ===")
df_prep = prepare_fraud_features(df)
train_df, test_df = train_test_split_fraud(df_prep, seed=123)

print(f"Train size: {len(train_df)} (fraud rate: {train_df['fraud_label'].mean():.4f})")
print(f"Test size: {len(test_df)} (fraud rate: {test_df['fraud_label'].mean():.4f})")

# Isolation Forest
iso_model = fit_isolation_forest(train_df, contamination=0.06, seed=123)
res_iso = evaluate_isolation_forest(iso_model, test_df)
print(f"Isolation Forest -> AUC-ROC: {res_iso['auc_roc']:.3f}, PR-AUC: {res_iso['pr_auc']:.3f}")

# Random Forest
rf_model = fit_supervised_baseline(train_df, seed=123)
res_rf = evaluate_supervised(rf_model, test_df)
print(f"Random Forest     -> AUC-ROC: {res_rf['auc_roc']:.3f}, PR-AUC: {res_rf['pr_auc']:.3f}")

# Feature importances
importances = pd.Series(
    rf_model.feature_importances_, 
    index=[c.replace("_code", "").replace("_norm", "") for c in train_df.columns if c.endswith(("_code", "_norm"))]
)
importances = importances / importances.sum()
print("\nTop 5 Feature Importances (Random Forest):")
print(importances.sort_values(ascending=False).head(5))

print("\n=== 4. GRAPH HOMOPHILY & ATTEMPTS ===")
y_all = df_prep["fraud_label"].values
p = y_all.mean()
random_homophily = p**2 + (1-p)**2
print(f"Random chance homophily reference: {random_homophily:.4f} (or {random_homophily:.3f})")

# RepNumber stats
rep_counts = df["RepNumber"].nunique()
print(f"\nRepNumber unique values: {rep_counts}, Avg rows per RepNumber: {len(df)/rep_counts:.1f}")

# Attempt 1: RepNumber + 2/3 profile attributes (Make, VehicleCategory, PolicyType)
print("\n--- Tentative 1 : RepNumber + 2/3 profile attributes ---")
feature_cols = [c for c in df_prep.columns if c.endswith(("_code", "_norm")) and c not in ["RepNumber_code"]]
graph_t1 = build_pyg_graph(df_prep, feature_cols, min_shared_attrs=2)
e1 = graph_t1.edge_index.numpy()
deg1 = graph_t1.num_edges / graph_t1.num_nodes
hom1 = (y_all[e1[0]] == y_all[e1[1]]).mean() if graph_t1.num_edges > 0 else 0
print(f"Edges: {graph_t1.num_edges}, Avg degree: {deg1:.1f}, Homophily: {hom1:.3f}")

# Attempt 2: Generic profile 6 strict attributes (Make, VehicleCategory, PolicyType, AccidentArea, AgeOfVehicle, BasePolicy)
print("\n--- Tentative 2 : Generic profile 6 strict attributes ---")
cols_t2 = ["Make", "VehicleCategory", "PolicyType", "AccidentArea", "AgeOfVehicle", "BasePolicy"]
e2 = build_edge_index(df, similarity_cols=cols_t2, min_shared_attrs=len(cols_t2))
deg2 = e2.shape[1] / len(df)
hom2 = (y_all[e2[0].numpy()] == y_all[e2[1].numpy()]).mean() if e2.shape[1] > 0 else 0
print(f"Edges: {e2.shape[1]}, Avg degree: {deg2:.1f}, Homophily: {hom2:.3f}")

# Attempt 3: Targeted profile 5 strict attributes (Fault, AddressChange_Claim, Days_Policy_Claim, PolicyType, BasePolicy)
print("\n--- Tentative 3 : Targeted profile 5 strict attributes ---")
cols_t3 = ["Fault", "AddressChange_Claim", "Days_Policy_Claim", "PolicyType", "BasePolicy"]
e3 = build_edge_index(df, similarity_cols=cols_t3, min_shared_attrs=len(cols_t3))
deg3 = e3.shape[1] / len(df)
hom3 = (y_all[e3[0].numpy()] == y_all[e3[1].numpy()]).mean() if e3.shape[1] > 0 else 0
print(f"Edges: {e3.shape[1]}, Avg degree: {deg3:.2f}, Homophily: {hom3:.3f}")

# Attempt 4: Shared rare value (threshold < 10%)
print("\n--- Tentative 4 : Shared rare value (<10%) ---")
cols_t4 = ["Fault", "AddressChange_Claim", "Days_Policy_Claim", "Days_Policy_Accident", "PastNumberOfClaims"]
e4 = build_edge_index_rare_shared(df, cols_t4, rarity_threshold=0.10)
deg4 = e4.shape[1] / len(df)
hom4 = (y_all[e4[0].numpy()] == y_all[e4[1].numpy()]).mean() if e4.shape[1] > 0 else 0
print(f"Edges: {e4.shape[1]}, Avg degree: {deg4:.1f}, Homophily: {hom4:.3f}")
