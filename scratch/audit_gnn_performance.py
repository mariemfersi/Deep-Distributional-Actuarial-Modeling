import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from pathlib import Path
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.fraud.data import load_fraud_data, prepare_fraud_features, train_test_split_fraud
from src.fraud.graph import build_edge_index_repnumber, build_edge_index_similarity, build_edge_index_rare_shared

print("=== 1. DATA PREPARATION FOR GNN AUDIT ===")
df_raw = load_fraud_data()
df_prep = prepare_fraud_features(df_raw)

# Splitting train / test (80/20 seed 123)
train_df, test_df = train_test_split_fraud(df_prep, seed=123)

print(f"Total dataset: {len(df_prep)} rows | Fraud: {df_prep['fraud_label'].sum()} ({df_prep['fraud_label'].mean():.4f})")
print(f"Train set:     {len(train_df)} rows | Fraud: {train_df['fraud_label'].sum()} ({train_df['fraud_label'].mean():.4f})")
print(f"Test set:      {len(test_df)} rows | Fraud: {test_df['fraud_label'].sum()} ({test_df['fraud_label'].mean():.4f})")

feature_cols = [c for c in df_prep.columns if c.endswith(("_code", "_norm")) and c not in ["RepNumber_code"]]

# Build full PyG Data object
X_all = torch.tensor(df_prep[feature_cols].values, dtype=torch.float32)
y_all = torch.tensor(df_prep["fraud_label"].values, dtype=torch.long)

# Train mask & Test mask
n_total = len(df_prep)
n_train = len(train_df)
n_test = len(test_df)

train_mask = torch.zeros(n_total, dtype=torch.bool)
test_mask = torch.zeros(n_total, dtype=torch.bool)

train_mask[:n_train] = True
test_mask[n_train:] = True

print(f"\nTrain mask sum: {train_mask.sum().item()}, Test mask sum: {test_mask.sum().item()}")

# Build edge indices for different graph strategies
edges_rep = build_edge_index_repnumber(df_prep, min_shared_attrs=2)
edges_sim_gen = build_edge_index_similarity(df_prep, similarity_cols=["Make", "VehicleCategory", "PolicyType", "AccidentArea", "AgeOfVehicle", "BasePolicy"])
edges_sim_tgt = build_edge_index_similarity(df_prep, similarity_cols=["Fault", "AddressChange_Claim", "Days_Policy_Claim", "PolicyType", "BasePolicy"])
edges_rare = build_edge_index_rare_shared(df_prep, cols=["Fault", "AddressChange_Claim", "Days_Policy_Claim", "Days_Policy_Accident", "PastNumberOfClaims"], rarity_threshold=0.10)

class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        out = self.fc(h).squeeze(-1)
        return out

print("\n=== 2. TRAINING GRAPHSAGE ON EACH GRAPH STRATEGY ===")

strategies = {
    "Tentative 1 (RepNumber + 2/3 profil)": edges_rep,
    "Tentative 2 (Profil générique 6 stricts)": edges_sim_gen,
    "Tentative 3 (Profil ciblé 5 stricts)": edges_sim_tgt,
    "Tentative 4 (Valeur rare partagée <10%)": edges_rare,
}

for name, edge_index in strategies.items():
    if edge_index.shape[1] == 0:
        print(f"\n{name} -> No edges built.")
        continue

    # Train model
    torch.manual_seed(123)
    model = GraphSAGEModel(in_dim=X_all.shape[1], hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Class weight to handle imbalance
    pos_weight = torch.tensor([(1 - train_df['fraud_label'].mean()) / train_df['fraud_label'].mean()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(150):
        optimizer.zero_grad()
        logits = model(X_all, edge_index)
        loss = criterion(logits[train_mask], y_all[train_mask].float())
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_all, edge_index)[test_mask]
        test_probs = torch.sigmoid(test_logits).numpy()
        test_y = y_all[test_mask].numpy()

    auc = roc_auc_score(test_y, test_probs)
    pr_auc = average_precision_score(test_y, test_probs)
    print(f"\n{name}:")
    print(f"  Edges count: {edge_index.shape[1]}")
    print(f"  AUC-ROC:  {auc:.3f}")
    print(f"  PR-AUC:   {pr_auc:.3f}")
    print(f"  Test prevalence: {test_y.mean():.4f} ({test_y.sum()}/{len(test_y)})")
