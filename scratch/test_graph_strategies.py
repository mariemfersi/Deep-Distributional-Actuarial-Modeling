import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.fraud.data import load_fraud_data, prepare_fraud_features

df = load_fraud_data()
df_prep = prepare_fraud_features(df)
y = df_prep["fraud_label"].values
p = y.mean()
random_homophily = p**2 + (1-p)**2

print(f"Base prevalence: {p:.4f}")
print(f"Random chance homophily reference: {random_homophily:.4f} (or {random_homophily:.3f})")

# Attempt 1: RepNumber + 2/3 profile attributes (Make, VehicleCategory, PolicyType)
def build_attempt1_edges(df, min_shared_attrs=2):
    edges = []
    similarity_cols = ["Make", "VehicleCategory", "PolicyType"]
    for rep, group in df.groupby("RepNumber"):
        if len(group) < 2:
            continue
        indices = group.index.tolist()
        # To make it fast, compare against group
        vals = group[similarity_cols].values
        n = len(indices)
        for i in range(n):
            for j in range(i + 1, n):
                shared = (vals[i] == vals[j]).sum()
                if shared >= min_shared_attrs:
                    edges.append((indices[i], indices[j]))
                    edges.append((indices[j], indices[i]))
    return np.array(edges).T if len(edges) > 0 else np.empty((2, 0))

e1 = build_attempt1_edges(df_prep, min_shared_attrs=2)
deg1 = e1.shape[1] / len(df_prep)
hom1 = (y[e1[0]] == y[e1[1]]).mean() if e1.shape[1] > 0 else 0
print(f"Attempt 1 -> Edges: {e1.shape[1]}, Avg degree: {deg1:.1f}, Homophily: {hom1:.4f} ({hom1:.3f})")

# Attempt 2: Generic profile, 6 strict attributes (Make, VehicleCategory, PolicyType, AccidentArea, AgeOfVehicle, BasePolicy)
from collections import defaultdict

def build_attempt2_edges(df, similarity_cols):
    edges = []
    groups = defaultdict(list)
    for idx, row in df[similarity_cols].iterrows():
        key = tuple(row.values)
        groups[key].append(idx)
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                edges.append((indices[i], indices[j]))
                edges.append((indices[j], indices[i]))
    return np.array(edges).T if len(edges) > 0 else np.empty((2, 0))

cols_t2 = ["Make", "VehicleCategory", "PolicyType", "AccidentArea", "AgeOfVehicle", "BasePolicy"]
e2 = build_attempt2_edges(df, cols_t2)
deg2 = e2.shape[1] / len(df)
hom2 = (y[e2[0]] == y[e2[1]]).mean() if e2.shape[1] > 0 else 0
print(f"Attempt 2 -> Edges: {e2.shape[1]}, Avg degree: {deg2:.1f}, Homophily: {hom2:.4f} ({hom2:.3f})")

# Attempt 3: Targeted profile, 5 strict attributes (Fault, AddressChange_Claim, Days_Policy_Claim, PolicyType, BasePolicy)
cols_t3 = ["Fault", "AddressChange_Claim", "Days_Policy_Claim", "PolicyType", "BasePolicy"]
e3 = build_attempt2_edges(df, cols_t3)
deg3 = e3.shape[1] / len(df)
hom3 = (y[e3[0]] == y[e3[1]]).mean() if e3.shape[1] > 0 else 0
print(f"Attempt 3 -> Edges: {e3.shape[1]}, Avg degree: {deg3:.2f}, Homophily: {hom3:.4f} ({hom3:.3f})")

# Attempt 4: Shared rare value (<10%)
from src.fraud.graph import build_edge_index_rare_shared
cols_t4 = ["Fault", "AddressChange_Claim", "Days_Policy_Claim", "Days_Policy_Accident", "PastNumberOfClaims"]
e4_tensor = build_edge_index_rare_shared(df, cols_t4, rarity_threshold=0.10)
e4 = e4_tensor.numpy()
deg4 = e4.shape[1] / len(df)
hom4 = (y[e4[0]] == y[e4[1]]).mean() if e4.shape[1] > 0 else 0
print(f"Attempt 4 -> Edges: {e4.shape[1]}, Avg degree: {deg4:.1f}, Homophily: {hom4:.4f} ({hom4:.3f})")
