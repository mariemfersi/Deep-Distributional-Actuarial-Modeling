"""Prototype : GNNExplainer réel appliqué au module fraude (section 7.3)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

from src.fraud.data import load_fraud_data, prepare_fraud_features, train_test_split_fraud
from src.fraud.graph import build_edge_index_similarity, build_edge_index_rare_shared


class GraphSAGEModel(nn.Module):
    """Réplique exacte du modèle utilisé dans scratch/audit_gnn_performance.py."""
    def __init__(self, in_dim, hidden_dim=32, out_dim=1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        out = self.fc(h).squeeze(-1)
        return out


def main():
    torch.manual_seed(123)

    # --- Data ---
    df_raw = load_fraud_data()
    df_prep = prepare_fraud_features(df_raw)
    train_df, test_df = train_test_split_fraud(df_prep, seed=123)
    feature_cols = [c for c in df_prep.columns if c.endswith(("_code", "_norm")) and c != "RepNumber_code"]

    X_all = torch.tensor(df_prep[feature_cols].values, dtype=torch.float32)
    y_all = torch.tensor(df_prep["fraud_label"].values, dtype=torch.long)
    n_total = len(df_prep)
    n_train = len(train_df)
    train_mask = torch.zeros(n_total, dtype=torch.bool)
    train_mask[:n_train] = True

    # Stratégie de graphe "Tentative 3" (profil ciblé), la plus significative
    sim_cols = ["Fault", "AddressChange_Claim", "Days_Policy_Claim", "PolicyType", "BasePolicy"]
    edge_index = build_edge_index_similarity(df_prep, similarity_cols=sim_cols)
    print(f"Graphe : {edge_index.shape[1]} arêtes (directed symétriques)")

    # --- Training ---
    model = GraphSAGEModel(in_dim=X_all.shape[1], hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    pos_weight = torch.tensor([(1 - train_df['fraud_label'].mean()) / train_df['fraud_label'].mean()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(150):
        optimizer.zero_grad()
        logits = model(X_all, edge_index)
        loss = criterion(logits[train_mask], y_all[train_mask].float())
        loss.backward()
        optimizer.step()
    print(f"Entraînement terminé — loss finale = {loss.item():.4f}")

    # --- Sélection du dossier cible (test, prédit fraude, réellement frauduleux) ---
    model.eval()
    with torch.no_grad():
        all_logits = model(X_all, edge_index)
    test_start = n_train
    probs = torch.sigmoid(all_logits[test_start:]).numpy()
    y_test = y_all[test_start:].numpy()

    from sklearn.metrics import roc_auc_score, average_precision_score
    print(f"AUC-ROC test = {roc_auc_score(y_test, probs):.3f} | PR-AUC = {average_precision_score(y_test, probs):.3f}")

    # Candidats : vrais positifs de test les plus confiants
    tp = [(k, p) for k, p in enumerate(probs) if y_test[k] == 1]
    tp_sorted = sorted(tp, key=lambda x: -x[1])
    print(f"Vrais positifs de test : {len(tp)} | top-3 scores : {[round(p,3) for _,p in tp_sorted[:3]]}")

    for k, p in tp_sorted[:3]:
        node_idx = test_start + k
        # Taille du voisinage 2-hop
        from torch_geometric.utils import k_hop_subgraph
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(node_idx, 2, edge_index)
        print(f"node_idx={node_idx} nom={sub_nodes[0]} prob={p:.3f} 2-hop={len(sub_nodes)} nodes {sub_edge_index.shape[1]} edges")
        if len(sub_nodes) <= 40:
            target_idx = node_idx
            break
    else:
        target_idx = test_start + tp_sorted[0][0]

    print(f"\n=== GNNExplainer sur node_idx = {target_idx} ===")
    explainer = Explainer(
        model,
        algorithm=GNNExplainer(epochs=200, lr=0.01, edge_size=0.005, edge_ent=1.0, node_feat_ent=0.1),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(mode="binary_classification", task_level="node", return_type="raw"),
    )
    explanation = explainer(X_all, edge_index, index=target_idx)
    print("node_mask shape:", explanation.node_mask.shape)
    print("edge_mask shape:", explanation.edge_mask.shape)

    em = explanation.edge_mask.detach().numpy()
    nm = explanation.node_mask.detach().numpy()
    print(f"edge_mask: min={em.min():.3f} max={em.max():.3f} mean={em.mean():.3f} n_edges={len(em)}")
    print(f"node_mask: min={nm.min():.3f} max={nm.max():.3f}")
    top_k = nm.mean(axis=0) if nm.ndim == 2 else nm
    top_idx = top_k.argsort()[::-1][:8]
    feat_names = [feature_cols[i].removesuffix("_code").removesuffix("_norm") for i in top_idx]
    print("Top-8 features :", [(feat_names[j], round(float(top_k[top_idx[j]]), 3)) for j in range(8)])


if __name__ == "__main__":
    main()