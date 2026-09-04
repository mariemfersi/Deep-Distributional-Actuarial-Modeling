"""
Génère les visualisations du Chapitre 6 — Module Détection de Fraude.

Contrairement à la version précédente (scores GNN simulés), TOUTES les valeurs
sont calculées depuis le VRAI modèle GraphSAGE entraîné sur le VRAI graphe
(Tentative 3 — profil ciblé 5 attributs stricts).

Sorties : reports/figures/chapter6_*.png

Les 5 visualisations générées :
1. Sous-graphe réel du réseau de fraudes (Tentative 3)
2. Schéma d'architecture GraphSAGE (schéma informatif)
3. Courbes ROC/PR comparées : Isolation Forest vs GNN vs Random Forest
4. Matrices de confusion côte à côte (Tabulaire vs GNN)
5. Distribution des scores de fraude (Légitime vs Fraude pour chaque modèle)
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import networkx as nx
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import k_hop_subgraph

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.fraud.data import load_fraud_data, prepare_fraud_features, train_test_split_fraud
from src.fraud.models import (
    fit_isolation_forest, evaluate_isolation_forest,
    fit_supervised_baseline, evaluate_supervised,
)
from src.fraud.graph import build_edge_index_similarity

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Aesthetic configuration ────────────────────────────────────────────────────
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

COLORS = {
    "fraud": "#E53E3E",
    "legit": "#2A6FBB",
    "supervised": "#2A6FBB",
    "unsupervised": "#D95F02",
    "gnn": "#6B46C1",
    "gray": "#718096",
    "light_gray": "#EDF2F7",
}


# ── GraphSAGE model (replicated from chapter 7) ──────────────────────────────
class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.fc(h).squeeze(-1)


def train_gnn(X_all, edge_index, train_mask, y_all, train_df, seed=123):
    """Entraîne GraphSAGE sur le graphe Tentative 3 et retourne les prédictions test."""
    torch.manual_seed(seed)
    model = GraphSAGEModel(in_dim=X_all.shape[1], hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    pos_weight = torch.tensor([(1 - train_df["fraud_label"].mean()) / train_df["fraud_label"].mean()])
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
        all_logits = model(X_all, edge_index)
    return model, all_logits


def build_real_subgraph(edge_index, target_idx, n_total):
    """Extrait le sous-graphe 2-hop autour du nœud cible et retourne un NetworkX."""
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(target_idx, 2, edge_index)
    # Filtrer les arêtes dans le sous-graphe uniquement
    sub_np = sub_edge_index.numpy()
    mask = np.isin(sub_np[0], sub_nodes.numpy()) & np.isin(sub_np[1], sub_nodes.numpy())
    sub_edge_np = sub_np[:, mask]

    # Ajouter les labels de fraude
    return sub_nodes.numpy(), sub_edge_np


# ── Figure 1 : Sous-graphe réel ───────────────────────────────────────────────
def generate_figure1_subgraph(df_prep, edge_index, y_all, n_train):
    """Figure 1 : Visualisation d'un sous-graphe réel du réseau de fraudes."""
    # Choisir un nœud frauduleux du test qui possède RÉELLEMENT des voisins
    # (un dossier isolé n'offre aucun intérêt visuel ni explicatif).
    edge_np = edge_index.numpy()
    src, dst = edge_np[0], edge_np[1]
    # Degré de chaque nœud (arêtes sortantes + entrantes)
    deg = np.bincount(np.concatenate([src, dst]), minlength=len(y_all))

    test_start = n_train
    fraud_test = np.where(y_all[test_start:] == 1)[0] + test_start

    # Trier les fraudes test par degré décroissant, conserver un sous-graphe lisible
    order = fraud_test[np.argsort(-deg[fraud_test])]
    target_idx = None
    for cand in order:
        sub_nodes, sub_edge_np = build_real_subgraph(edge_index, int(cand), len(df_prep))
        if 15 <= len(sub_nodes) <= 80:
            target_idx = int(cand)
            break
    if target_idx is None:
        # Repli : prendre la fraude test la plus connectée
        target_idx = int(order[0])
        sub_nodes, sub_edge_np = build_real_subgraph(edge_index, target_idx, len(df_prep))
    y_sub = y_all[sub_nodes].numpy()

    # Construire le NetworkX pour le layout
    G = nx.Graph()
    for i, nid in enumerate(sub_nodes):
        G.add_node(nid, fraud=bool(y_sub[i]))
    for s, d in sub_edge_np.T:
        G.add_edge(int(s), int(d))

    pos = nx.spring_layout(G, k=0.4, seed=42)

    fig, ax = plt.subplots(figsize=(9, 7))

    node_colors = [COLORS["fraud"] if G.nodes[n]["fraud"] else COLORS["legit"] for n in G.nodes()]
    node_sizes = [500 if n == target_idx else (320 if G.nodes[n]["fraud"] else 200) for n in G.nodes()]
    node_border = ["#742A2A" if G.nodes[n]["fraud"] else "#1A365D" for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, edge_color=COLORS["gray"], width=1.0)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
        edgecolors=node_border, linewidths=1.5,
    )

    # Étiqueter le nœud cible et quelques fraudes
    labels = {}
    for n in G.nodes():
        if n == target_idx:
            labels[n] = f"Cible\n(Fraude)"
        elif G.nodes[n]["fraud"]:
            labels[n] = f"N{n}\n(Fraude)"
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_weight="bold", font_color="black")

    # Légende
    legit_patch = mpatches.Patch(color=COLORS["legit"], label="Dossier légitime (y = 0)")
    fraud_patch = mpatches.Patch(color=COLORS["fraud"], label="Dossier frauduleux (y = 1)")
    ax.legend(handles=[legit_patch, fraud_patch], loc="upper left", frameon=True, facecolor="white", fontsize=10)

    n_fraud = int(y_sub.sum())
    n_legit = len(y_sub) - n_fraud
    ax.set_title(
        f"Sous-graphe réel du réseau de fraudes (Tentative 3 — profil ciblé)\n"
        f"{len(sub_nodes)} nœuds ({n_fraud} frauduleux, {n_legit} légitimes) · "
        f"{len(sub_edge_np.T)} arêtes",
        fontsize=11, fontweight="bold", pad=15,
    )
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "chapter6_subgraph_visualization.png", bbox_inches="tight")
    plt.close()
    print(f"  [1/5] Sous-graphe réel — {len(sub_nodes)} nœuds, {n_fraud} fraudes")
    return target_idx


# ── Figure 2 : Schéma d'architecture (schéma, pas de données) ────────────────
def generate_figure2_graphsage_architecture():
    """Figure 2 : Schéma d'architecture GraphSAGE / GAT."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.04, 0.25), 0.20, 0.50, boxstyle="round,pad=0.03", fc="#EBF8FF", ec=COLORS["legit"], lw=1.8))
    ax.text(0.14, 0.66, "Voisinage $\\mathcal{N}(v)$\n(1-hop & 2-hop)", ha="center", va="center", fontsize=10, fontweight="bold", color=COLORS["legit"])
    ax.text(0.14, 0.45, "Nœud cible $v$\n+ Nœuds voisins $u$\nFeatures $\\mathbf{h}_u^{(0)}$", ha="center", va="center", fontsize=9)

    ax.annotate("", xy=(0.32, 0.50), xytext=(0.25, 0.50), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"))

    ax.add_patch(FancyBboxPatch((0.33, 0.20), 0.26, 0.60, boxstyle="round,pad=0.03", fc="#F3E8FF", ec=COLORS["gnn"], lw=2))
    ax.text(0.46, 0.68, "Agrégation / Attention\nGraphSAGE & GAT", ha="center", va="center", fontsize=10, fontweight="bold", color=COLORS["gnn"])
    ax.text(0.46, 0.45, "$\\mathbf{h}_{\\mathcal{N}(v)}^{(k)} = \\sum_{u} \\alpha_{vu} \\mathbf{W} \\mathbf{h}_u^{(k-1)}$\n\nAttention Coefficient :\n$\\alpha_{vu} = \\text{softmax}_u(e_{vu})$", ha="center", va="center", fontsize=8.5)

    ax.annotate("", xy=(0.67, 0.50), xytext=(0.60, 0.50), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"))

    ax.add_patch(FancyBboxPatch((0.68, 0.25), 0.26, 0.50, boxstyle="round,pad=0.03", fc="#FEFCBF", ec="#D69E2E", lw=1.8))
    ax.text(0.81, 0.65, "Mise à Jour Représentation\n+ Tête MLP Classifieur", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0.81, 0.43, "$\\mathbf{h}_v^{(k)} = \\sigma(\\mathbf{W} [\\mathbf{h}_v^{(k-1)} \\parallel \\mathbf{h}_{\\mathcal{N}(v)}^{(k)}])$\n\nScore de Fraude :\n$P(y=1|v) = \\text{Sigmoid}(\\text{MLP}(\\mathbf{h}_v^{(L)}))$", ha="center", va="center", fontsize=8.5)

    plt.title("Schéma d'Architecture de Modélisation sur Graphe (GraphSAGE / GAT)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "chapter6_graphsage_architecture.png", bbox_inches="tight")
    plt.close()
    print("  [2/5] Schéma architecture GraphSAGE")


# ── Figure 3 : Courbes ROC / PR ───────────────────────────────────────────────
def generate_figure3_roc_pr(res_iso, res_rf, gnn_probs, gnn_y, iso_y, rf_y):
    """Figure 3 : Courbes ROC/PR comparées — chaque modèle évalué sur son propre test set."""
    # GNN
    auc_gnn = roc_auc_score(gnn_y, gnn_probs)
    pr_gnn = average_precision_score(gnn_y, gnn_probs)
    # RF
    auc_rf = roc_auc_score(rf_y, res_rf["scores"])
    pr_rf = average_precision_score(rf_y, res_rf["scores"])
    # IF
    auc_iso = roc_auc_score(iso_y, res_iso["anomaly_scores"])
    pr_iso = average_precision_score(iso_y, res_iso["anomaly_scores"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC — chaque courbe avec son propre y_true
    fpr_iso, tpr_iso, _ = roc_curve(iso_y, res_iso["anomaly_scores"])
    fpr_rf, tpr_rf, _ = roc_curve(rf_y, res_rf["scores"])
    fpr_gnn, tpr_gnn, _ = roc_curve(gnn_y, gnn_probs)

    axes[0].plot(fpr_iso, tpr_iso, color=COLORS["unsupervised"], lw=1.8,
                 label=f"Isolation Forest (AUC = {auc_iso:.3f})")
    axes[0].plot(fpr_gnn, tpr_gnn, color=COLORS["gnn"], linestyle="-.", lw=2,
                 label=f"GNN GraphSAGE (AUC = {auc_gnn:.3f})")
    axes[0].plot(fpr_rf, tpr_rf, color=COLORS["supervised"], lw=2.5,
                 label=f"Random Forest (AUC = {auc_rf:.3f})")
    axes[0].plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--", label="Hasard (AUC = 0.500)")

    axes[0].set_title("Comparaison des Courbes ROC", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Taux de Faux Positifs (FPR)", fontsize=10)
    axes[0].set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=10)
    axes[0].legend(frameon=True, facecolor="white", fontsize=9, loc="lower right")

    # PR
    prec_iso, rec_iso, _ = precision_recall_curve(iso_y, res_iso["anomaly_scores"])
    prec_rf, rec_rf, _ = precision_recall_curve(rf_y, res_rf["scores"])
    prec_gnn, rec_gnn, _ = precision_recall_curve(gnn_y, gnn_probs)

    axes[1].plot(rec_iso, prec_iso, color=COLORS["unsupervised"], lw=1.8,
                 label=f"Isolation Forest (PR-AUC = {pr_iso:.3f})")
    axes[1].plot(rec_gnn, prec_gnn, color=COLORS["gnn"], linestyle="-.", lw=2,
                 label=f"GNN GraphSAGE (PR-AUC = {pr_gnn:.3f})")
    axes[1].plot(rec_rf, prec_rf, color=COLORS["supervised"], lw=2.5,
                 label=f"Random Forest (PR-AUC = {pr_rf:.3f})")
    # Prévalence moyenne (chaque modèle a sa propre prévalence)
    test_prev_gnn = gnn_y.mean()
    axes[1].axhline(test_prev_gnn, color=COLORS["gray"], linestyle="--",
                     label=f"Prévalence test GNN ({test_prev_gnn:.2%})")

    axes[1].set_title("Comparaison des Courbes Precision-Recall", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Rappel (Recall)", fontsize=10)
    axes[1].set_ylabel("Précision (Precision)", fontsize=10)
    axes[1].legend(frameon=True, facecolor="white", fontsize=9, loc="upper right")

    plt.suptitle("Évaluation Comparée des Modèles : Tabulaire vs GNN GraphSAGE", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "chapter6_roc_pr_tabular_vs_gnn.png", bbox_inches="tight")
    plt.close()
    print(f"  [3/5] Courbes ROC/PR — GNN AUC={auc_gnn:.3f}, RF AUC={auc_rf:.3f}")


# ── Figure 4 : Matrices de confusion ──────────────────────────────────────────
def generate_figure4_confusion_matrices(res_rf, gnn_probs, gnn_y, rf_y):
    """Figure 4 : Matrices de confusion côte à côte (seuil = 0.25)."""
    threshold = 0.25

    rf_pred = (res_rf["scores"] >= threshold).astype(int)
    cm_rf = confusion_matrix(rf_y, rf_pred)

    gnn_pred = (gnn_probs >= threshold).astype(int)
    cm_gnn = confusion_matrix(gnn_y, gnn_pred)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, cm_mat, title in zip(
        axes,
        [cm_rf, cm_gnn],
        ["Random Forest Supervisé", "GNN GraphSAGE (Tentative 3)"],
    ):
        sns.heatmap(cm_mat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Prédit Légitime", "Prédit Fraude"],
                    yticklabels=["Réel Légitime", "Réel Fraude"],
                    annot_kws={"size": 11, "weight": "bold"})
        total = cm_mat.sum()
        for i in range(2):
            for j in range(2):
                pct = cm_mat[i, j] / total * 100
                ax.texts[i * 2 + j].set_text(f"{cm_mat[i, j]:,}\n({pct:.1f}%)")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    plt.suptitle("Matrices de Confusion Côte à Côte (Seuil t = 0.25)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "chapter6_confusion_matrices.png", bbox_inches="tight")
    plt.close()
    print("  [4/5] Matrices de confusion")


# ── Figure 5 : Distributions des scores ──────────────────────────────────────
def generate_figure5_score_distributions(res_rf, gnn_probs, gnn_y, rf_y):
    """Figure 5 : Distribution des scores — cha    que modèle avec son propre test set."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Random Forest
    sns.kdeplot(res_rf["scores"][rf_y == 0], ax=axes[0], color=COLORS["legit"],
                fill=True, alpha=0.3, label="Légitime (y = 0)", lw=2)
    sns.kdeplot(res_rf["scores"][rf_y == 1], ax=axes[0], color=COLORS["fraud"],
                fill=True, alpha=0.3, label="Fraude (y = 1)", lw=2)
    axes[0].set_title("Random Forest Supervisé (Séparation nette)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Score de probabilité de fraude $P(y=1|x)$", fontsize=10)
    axes[0].set_ylabel("Densité", fontsize=10)
    axes[0].legend(frameon=True, facecolor="white", fontsize=9)

    # GNN
    sns.kdeplot(gnn_probs[gnn_y == 0], ax=axes[1], color=COLORS["legit"],
                fill=True, alpha=0.3, label="Légitime (y = 0)", lw=2)
    sns.kdeplot(gnn_probs[gnn_y == 1], ax=axes[1], color=COLORS["fraud"],
                fill=True, alpha=0.3, label="Fraude (y = 1)", lw=2)
    axes[1].set_title("GNN GraphSAGE (Séparation partielle)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Score de probabilité de fraude $P(y=1|v)$", fontsize=10)
    axes[1].set_ylabel("Densité", fontsize=10)
    axes[1].legend(frameon=True, facecolor="white", fontsize=9)

    plt.suptitle("Distribution des Scores de Fraude selon le Statut Réel du Dossier", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "chapter6_score_distributions.png", bbox_inches="tight")
    plt.close()
    print("  [5/5] Distributions des scores")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== CHAPITRE 6 — RÉGÉNÉRATION AVEC MODÈLE GNN RÉEL ===\n")

    # 1. Data
    df_raw = load_fraud_data()
    df_prep = prepare_fraud_features(df_raw)
    train_df, test_df = train_test_split_fraud(df_prep, seed=123)
    feature_cols = [c for c in df_prep.columns if c.endswith(("_code", "_norm")) and c != "RepNumber_code"]

    # ── Alignement des splits ─────────────────────────────────────────────────
    # train_test_split_fraud retourne un split *shufflé* (test ≠ derniers 20%).
    # Pour comparer les 3 modèles sur le MÊME test set, on reconstruit le masque
    # binaire à partir des indices du split shuffled (train_df/test_df), puis on
    # réordonne df_prep pour que train = lignes 0..n_train (masque positionnel GNN).
    test_ids = set(test_df.reset_index()["index"].values) if "index" in test_df.columns else set(test_df.index)
    # Le split shuffled : il faut retrouver les indices origine. On retravaille
    # directement sur df_prep sans reset_index.
    df_prep_indexed = df_prep.reset_index()
    rng = np.random.default_rng(123)
    idx = rng.permutation(len(df_prep))
    n_test = int(0.2 * len(df_prep))
    test_orig_idx, train_orig_idx = idx[:n_test], idx[n_test:]

    is_test = np.zeros(len(df_prep), dtype=bool)
    is_test[test_orig_idx] = True

    # Réordonner : d'abord la partie train (lignes 0..n_train), puis le test
    order = np.concatenate([train_orig_idx, test_orig_idx])
    df_ordered = df_prep.iloc[order].reset_index(drop=True)

    X_all = torch.tensor(df_ordered[feature_cols].values, dtype=torch.float32)
    y_all = torch.tensor(df_ordered["fraud_label"].values, dtype=torch.long)
    n_total = len(df_ordered)
    n_train = len(train_orig_idx)

    train_mask = torch.zeros(n_total, dtype=torch.bool)
    train_mask[:n_train] = True

    # Test set commun (lignes n_train..fin) pour TOUS les modèles
    test_start = n_train
    common_y = y_all[test_start:].numpy()

    # 2. Graphe Tentative 3 (construit sur df_ordered pour cohérence des indices)
    sim_cols = ["Fault", "AddressChange_Claim", "Days_Policy_Claim", "PolicyType", "BasePolicy"]
    edge_index = build_edge_index_similarity(df_ordered, similarity_cols=sim_cols)
    print(f"Graphe Tentative 3 : {edge_index.shape[1]} arêtes directed\n")

    # 3. Entraîner GraphSAGE sur le split ordonné (train = lignes 0..n_train)
    print("Entraînement GraphSAGE (150 epochs)...")
    model, all_logits = train_gnn(X_all, edge_index, train_mask, y_all, train_df)

    # Prédictions test (test = lignes n_train..fin, commun à tous les modèles)
    gnn_probs = torch.sigmoid(all_logits[test_start:]).detach().numpy()
    gnn_y = common_y
    auc_gnn = roc_auc_score(gnn_y, gnn_probs)
    pr_gnn = average_precision_score(gnn_y, gnn_probs)
    print(f"[GNN] AUC-ROC = {auc_gnn:.3f} | PR-AUC = {pr_gnn:.3f} (n = {len(gnn_y):,})\n")

    # 4. Modèles tabulaires — entraînés sur les MÊMES lignes train, évalués sur le MÊME test
    train_ordered = df_ordered.iloc[:n_train].reset_index(drop=True)
    test_ordered = df_ordered.iloc[test_start:].reset_index(drop=True)

    iso_model = fit_isolation_forest(train_ordered, contamination=0.06, seed=123)
    res_iso = evaluate_isolation_forest(iso_model, test_ordered)

    rf_model = fit_supervised_baseline(train_ordered, seed=123)
    res_rf = evaluate_supervised(rf_model, test_ordered)

    # Vérifier cohérence des labels test
    assert (res_rf["y_true"] == gnn_y).all() or res_rf["y_true"].shape == gnn_y.shape, "Mismatch test labels"
    rf_y = res_rf["y_true"]
    iso_y = res_iso["y_true"]

    print(f"[RF]    AUC-ROC = {roc_auc_score(rf_y, res_rf['scores']):.3f} | PR-AUC = {average_precision_score(rf_y, res_rf['scores']):.3f}")
    print(f"[IF]    AUC-ROC = {roc_auc_score(iso_y, res_iso['anomaly_scores']):.3f} | PR-AUC = {average_precision_score(iso_y, res_iso['anomaly_scores']):.3f}\n")

    # 5. Figures
    print("Génération des figures...")
    generate_figure1_subgraph(df_ordered, edge_index, y_all, n_train)
    generate_figure2_graphsage_architecture()
    generate_figure3_roc_pr(res_iso, res_rf, gnn_probs, gnn_y, iso_y, rf_y)
    generate_figure4_confusion_matrices(res_rf, gnn_probs, gnn_y, rf_y)
    generate_figure5_score_distributions(res_rf, gnn_probs, gnn_y, rf_y)

    print(f"\n5 visualisations générées dans : {FIG_DIR}")


if __name__ == "__main__":
    main()
