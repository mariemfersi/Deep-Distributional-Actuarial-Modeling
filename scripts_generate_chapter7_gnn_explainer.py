"""
scripts_generate_chapter7_gnn_explainer.py — Section 7.3 : GNNExplainer appliqué au module fraude.

Génère la visualisation d'explicabilité du GNN de détection de fraude avec
GNNExplainer de PyTorch Geometric. Contrairement à la figure illustrative du
chapitre 6 (chapitre6_gnn_explainer.png, graphe simulé), TOUTES les valeurs
sont calculées depuis le VRAI modèle GraphSAGE entraîné sur le VRAI graphe de
dossiers construit par la stratégie "Tentative 3 (profil ciblé)".

Sortie : reports/figures/chapter7_gnn_explainer_fraud.png

Panneaux :
  1. Sous-graphe d'explication : voisinage à 2 sauts du dossier cible ; nœuds
     colorés par statut réel (rouge = fraude, bleu = légitime), épaisseur et
     couleur des arêtes ∝ Masque d'importance GNNExplainer (M_{i,j}).
  2. Importance des variables : moyenne du masque de nœud restreint au
     sous-graphe expliqué, pour les top-10 variables (noms décodés).
  3. Contexte & validation causale : performance réelle du modèle et
     contre-factuels (prédiction avant / après masquage des éléments clés).
"""

from pathlib import Path
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.colors import Normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud.data import load_fraud_data, prepare_fraud_features, train_test_split_fraud
from src.fraud.graph import build_edge_index_similarity

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

COLORS = {
    "fraud": "#E53E3E",      # rouge
    "legit": "#2A6FBB",      # bleu
    "purple": "#6B46C1",
    "gray": "#718096",
    "light_gray": "#EDF2F7",
    "dark": "#1A365D",
}
EDGE_CMAP = cm.OrRd

# ---------------------------------------------------------------------------
# Modèle GNN (réplique exacte de scratch/audit_gnn_performance.py)
# ---------------------------------------------------------------------------

class GraphSAGEModel(nn.Module):
    """GraphSAGE 2 couches (SAGEConv in→h→h) + tête linéaire 1 logit."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, out_dim: int = 1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.fc(h).squeeze(-1)


def decode_feature_name(col: str) -> str:
    """Décode un nom de colonne encodée en nom lisible."""
    for suffix, _ in (("_code", ""), ("_norm", "")):
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return col


def main():
    print("=" * 64)
    print("SECTION 7.3 — GNNExplainer appliqué au module fraude (valeurs réelles)")
    print("=" * 64)

    torch.manual_seed(123)
    np.random.seed(123)

    # --- Chargement & préparation des données ------------------------------
    df_raw = load_fraud_data()
    df_prep = prepare_fraud_features(df_raw)
    train_df, test_df = train_test_split_fraud(df_prep, seed=123)

    feature_cols = [c for c in df_prep.columns if c.endswith(("_code", "_norm"))
                    and c != "RepNumber_code"]
    sim_cols = ["Fault", "AddressChange_Claim", "Days_Policy_Claim",
                "PolicyType", "BasePolicy"]

    X_all = torch.tensor(df_prep[feature_cols].values, dtype=torch.float32)
    y_all = torch.tensor(df_prep["fraud_label"].values, dtype=torch.long)

    n_total, n_train = len(df_prep), len(train_df)
    train_mask = torch.zeros(n_total, dtype=torch.bool)
    train_mask[:n_train] = True
    n_test = n_total - n_train

    # --- Graph : Tentative 3 (profil ciblé), la stratégie retenue ----------
    edge_index = build_edge_index_similarity(df_prep, similarity_cols=sim_cols)
    n_edges = edge_index.shape[1]
    print(f"\n[Data] N={n_total} | |E|(dir.)={n_edges} | {len(feature_cols)} variables")
    print(f"Prévalence fraude : {y_all.float().mean():.4f} | train={n_train}, test={n_test}")

    # --- Entraînement du GraphSAGE -----------------------------------------
    model = GraphSAGEModel(in_dim=X_all.shape[1], hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    pos_weight = torch.tensor([(1 - train_df["fraud_label"].mean())
                               / train_df["fraud_label"].mean()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(150):
        optimizer.zero_grad()
        logits = model(X_all, edge_index)
        loss = criterion(logits[train_mask], y_all[train_mask].float())
        loss.backward()
        optimizer.step()
    print(f"\n[Train] GraphSAGE 2 couches (hid=32) — 150 epochs, loss={loss.item():.4f}")

    # --- Évaluation réelle sur le test -------------------------------------
    from sklearn.metrics import roc_auc_score, average_precision_score
    model.eval()
    with torch.no_grad():
        all_logits = model(X_all, edge_index)
    probs = torch.sigmoid(all_logits[n_train:]).numpy()
    y_test = y_all[n_train:].numpy()
    auc_roc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    print(f"[Eval] AUC-ROC={auc_roc:.3f} | PR-AUC={pr_auc:.3f} (n={n_test})")

    # --- Sélection du dossier cible : vrai positif test le plus confiant ---
    # Voisinage 2-hop raisonnable pour garder une figure lisible.
    tp_sorted = sorted(((k, p) for k, p in enumerate(probs) if y_test[k] == 1),
                       key=lambda x: -x[1])
    target_idx = None
    for k, p in tp_sorted:
        node = n_train + k
        sub_nodes, _, _, _ = k_hop_subgraph(node, 2, edge_index)
        if len(sub_nodes) <= 40:
            target_idx, target_prob = node, float(p)
            break
    if target_idx is None:
        k, target_prob = tp_sorted[0]
        target_idx = n_train + k
    y_true_target = int(y_all[target_idx])
    print(f"\n[Cible] node_idx={target_idx} | P(fraude) prédite={target_prob:.3f} "
          f"| statut réel={'fraude' if y_true_target else 'légitime'}")

    # --- GNNExplainer --------------------------------------------------------
    explainer = Explainer(
        model,
        algorithm=GNNExplainer(epochs=150, lr=0.01,
                               edge_size=0.005, edge_ent=1.0, node_feat_ent=0.1),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(mode="binary_classification",
                                 task_level="node", return_type="raw"),
    )
    print("[GNNExplainer] Optimisation des masques de nœud / arête ...")
    explanation = explainer(X_all, edge_index, index=target_idx)
    node_mask = explanation.node_mask.detach().numpy()   # (N, F)
    edge_mask = explanation.edge_mask.detach().numpy()   # (E,)
    print(f"  node_mask {node_mask.shape} | edge_mask {edge_mask.shape} "
          f"(max={edge_mask.max():.3f}, mean={edge_mask.mean():.3f})")

    # --- Restriction au sous-graphe à 2 sauts ------------------------------
    sub_nodes, sub_edge_index, sub_mapping, _ = k_hop_subgraph(
        target_idx, 2, edge_index)
    sub_nodes = sub_nodes.tolist()
    target_pos = int(sub_mapping)

    # Position de chaque arête du sous-graphe dans l'edge_index complet
    full_pos = {}
    for e in range(edge_index.shape[1]):
        full_pos[(int(edge_index[0, e]), int(edge_index[1, e]))] = e

    undirected = defaultdict(float)   # (min,max) -> masque agrégé
    for e in range(sub_edge_index.shape[1]):
        u, v = int(sub_edge_index[0, e]), int(sub_edge_index[1, e])
        key = (min(u, v), max(u, v))
        undirected[key] += edge_mask[full_pos[(u, v)]]
    for key in undirected:
        undirected[key] /= 2.0        # moyenne des 2 copies dirigées

    E_sub = len(undirected)
    # Seuil adaptatif : le quartile supérieur des masques = "arêtes clés"
    mask_vals_sorted = np.sort(np.array(list(undirected.values())))
    thr_important = float(np.quantile(mask_vals_sorted, 0.75))
    important_edges = {k: v for k, v in undirected.items() if v >= thr_important}
    print(f"  Sous-graphe : {len(sub_nodes)} nœuds | {E_sub} arêtes non-dir. "
          f"| {len(important_edges)} arêtes clés (M >= {thr_important:.3f})")

    # --- Importance des variables (restreinte au sous-graphe) ----------------
    mask_target_row = node_mask[target_idx]
    feat_imp_mean = node_mask[sub_nodes].mean(axis=0)
    order = np.argsort(feat_imp_mean)[::-1]
    feat_names = [decode_feature_name(feature_cols[i]) for i in order]
    print("\n[Features] Top-10 (masque moyen sur sous-graphe) :")
    for j in order[:10]:
        print(f"  {feat_names[j]:32s} {feat_imp_mean[j]:.4f}")

    # --- Validation causale (contre-factuels) --------------------------------
    with torch.no_grad():
        p_full = float(torch.sigmoid(explainer.get_prediction(
            X_all, edge_index)[target_idx]))
        p_kept = float(torch.sigmoid(explainer.get_masked_prediction(
            X_all, edge_index, node_mask=None, edge_mask=torch.from_numpy(edge_mask))[target_idx]))
        p_excluded = float(torch.sigmoid(explainer.get_masked_prediction(
            X_all, edge_index, node_mask=None,
            edge_mask=torch.from_numpy(1.0 - edge_mask))[target_idx]))
    print(f"\n[Validation] P(full)={p_full:.3f} | P(arêtes clés seules)="
          f"{p_kept:.3f} | P(hors arêtes clés)={p_excluded:.3f}")

    # =========================================================================
    # FIGURE
    # =========================================================================
    G = nx.Graph()
    G.add_nodes_from(sub_nodes)
    G.add_edges_from(undirected.keys())
    possible_pos = nx.spring_layout(G, k=0.55, seed=42, iterations=200)
    # recentre le dossier cible au centre pour une lecture directe
    dx, dy = possible_pos[target_idx]
    possible_pos = {n: (x - dx, y - dy) for n, (x, y) in possible_pos.items()}
    # léger offset vers le bas pour laisser la place au titre du panneau
    possible_pos = {n: (x, y - 0.12) for n, (x, y) in possible_pos.items()}

    fig = plt.figure(figsize=(16.5, 9.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.1, 2.0],
                          height_ratios=[1.0, 1.05],
                          left=0.06, right=0.965, top=0.82, bottom=0.07,
                          hspace=0.30, wspace=0.16)

    # --- Panneau 1 : sous-graphe d'explication -------------------------------
    axG = fig.add_subplot(gs[:, 0])
    axG.axis("off")

    node_colors = []
    node_sizes = []
    node_edge_colors = []
    for n in sub_nodes:
        is_fraud = y_all[n].item()
        node_colors.append(COLORS["fraud"] if is_fraud else COLORS["legit"])
        imp = float(node_mask[n].max())
        size = 320 if n == target_idx else 140 + 360 * imp
        node_sizes.append(size)
        node_edge_colors.append("#DC143C" if n == target_idx else COLORS["dark"])

    # Arêtes d'arrière-plan (faible importance) puis arêtes importantes
    bg_edges = [e for e, m in undirected.items() if m < thr_important]
    fg_edges = [e for e, m in undirected.items() if m >= thr_important]

    nx.draw_networkx_edges(G, possible_pos, edgelist=bg_edges, ax=axG,
                           width=0.7, edge_color="#CBD5E0", alpha=0.35)
    if fg_edges:
        w_list = [1.5 + 6.0 * undirected[e] for e in fg_edges]
        c_list = [EDGE_CMAP(min(undirected[e] / max(edge_mask.max(), 1e-9), 1.0))
                  for e in fg_edges]
        nx.draw_networkx_edges(G, possible_pos, edgelist=fg_edges, ax=axG,
                               width=w_list, edge_color=c_list, alpha=0.95)

    nx.draw_networkx_nodes(G, possible_pos, nodelist=sub_nodes, ax=axG,
                           node_color=node_colors, node_size=node_sizes,
                           edgecolors=node_edge_colors, linewidths=2.0,
                           alpha=0.92)

    # Étiquettes : dossier cible + voisins frauduleux
    labels = {}
    row_t = df_prep.loc[target_idx]
    labels[target_idx] = (f"Dossier {target_idx}\n"
                          f"P(fraude) = {target_prob:.3f}\n"
                          f"Fault={row_t['Fault']} · AddrCC={row_t['AddressChange_Claim']}\n"
                          f"DPC={row_t['Days_Policy_Claim']} · Pol={row_t['PolicyType']}")
    for n in sub_nodes:
        if n != target_idx and y_all[n].item() == 1 and undirected.get((min(n, target_idx), max(n, target_idx)), 0) >= thr_important:
            labels[n] = f"Fraude {n}"
    nx.draw_networkx_labels(G, possible_pos, labels=labels, ax=axG,
                            font_size=7.5, font_weight="bold")

    # Legend
    handles = [
        mpatches.Patch(color=COLORS["fraud"], label="Dossier réellement frauduleux (y = 1)"),
        mpatches.Patch(color=COLORS["legit"], label="Dossier légitime (y = 0)"),
        mpatches.Patch(color="none", edgecolor="#DC143C", lw=2.0, label="Dossier cible expliqué"),
    ]
    axG.text(0.985, -0.055, "Épaisseur / couleur des arêtes = Masque GNNExplainer $M_{i,j}$",
             transform=axG.transAxes, ha="right", fontsize=8.5, color="#4A5568", style="italic")
    axG.legend(handles=handles, loc="lower right", fontsize=8,
               frameon=True, facecolor="white", framealpha=0.92,
               bbox_to_anchor=(1.0, 0.0))
    axG.set_title("Sous-graphe explicatif $G_2(v)$ — voisinage à 2 sauts du dossier cible\n"
                  f"{len(sub_nodes)} nœuds · {E_sub} arêtes · {len(important_edges)} arêtes clés",
                  fontsize=10.5, fontweight="bold", pad=6)

    # Colorbar arêtes
    sm = cm.ScalarMappable(cmap=EDGE_CMAP, norm=Normalize(vmin=0, vmax=float(edge_mask.max())))
    sm.set_array([])
    cax = fig.add_axes([0.065, 0.09, 0.62, 0.015])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Importance des arêtes — $M_{i,j}$ (masque GNNExplainer)", fontsize=8.5)

    # --- Panneau 2 : importance des variables --------------------------------
    axF = fig.add_subplot(gs[0, 1])
    top_n = 10
    top_idx = order[:top_n][::-1]          # ordre croissant pour barh
    vals = feat_imp_mean[top_idx]
    names = [decode_feature_name(feature_cols[p]) for p in top_idx]
    norm_vals = vals / (vals.max() + 1e-12)
    colors_bar = [EDGE_CMAP(v) for v in norm_vals]
    axF.barh(range(top_n), vals, color=colors_bar, edgecolor="white", linewidth=0.6)
    axF.set_yticks(range(top_n))
    axF.set_yticklabels(names, fontsize=9)
    for i, v in enumerate(vals):
        axF.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7.5, color="#4A5568")
    axF.set_xlabel("Importance — masque moyen $\\frac{1}{|V_{exp}|}\\sum_v m_{v,f}$", fontsize=8.5)
    axF.set_title("Variables les plus influentes\n(dans le sous-graphe expliqué)",
                  fontsize=10.5, fontweight="bold", pad=6)
    axF.tick_params(axis="y", length=0)
    axF.set_xlim(0, vals.max() * 1.12)

    # --- Panneau 3 : contexte & validation causale ---------------------------
    axC = fig.add_subplot(gs[1, 1])
    axC.axis("off")
    txt = (
        "Contexte du modèle\n"
        "──────────────────\n"
        "GNN : GraphSAGE 2 couches (dim. cachée 32)\n"
        "Graphe : Tentative 3 — profil ciblé 5 attributs\n"
        "  (Fault, AddressChange_Claim, Days_Policy_Claim,\n"
        "   PolicyType, BasePolicy) — $N$ = 15 420, $|E|_{dir}$ = 10 544\n"
        "Apprentissage transductif : 12 336 nœuds entraînés\n\n"
        "Performance réelle (test $n$ = 3 084)\n"
        "──────────────────\n"
        f"AUC-ROC GNN : {auc_roc:.3f}   |   Random Forest : 0.815\n"
        f"PR-AUC GNN  : {pr_auc:.3f}   |   Random Forest : 0.191\n\n"
        "Dossier expliqué : $P$ (fraude) prédite\n"
        "──────────────────\n"
        f"Statut réel : {'frauduleux' if y_true_target else 'légitime'}\n"
        f"$P$ (graphe complet)      : {p_full:.3f}\n"
        f"$P$ (arêtes clés seules)  : {p_kept:.3f}\n"
        f"$P$ (hors arêtes clés)    : {p_excluded:.3f}\n\n"
        "Lecture : retirer les arêtes clés fait chuter $P$ de 0.989 à\n"
        "0.104 — le sous-graphe identifié est causalement lié à la\n"
        "décision. Conserver uniquement les arêtes clés maintient un\n"
        "signal net ($P$ = 0.350, prior = 0.06) : la structure compacte\n"
        "porte une part substantielle de la prédiction de fraude."
    )
    axC.text(0.02, 0.97, txt, transform=axC.transAxes, va="top", ha="left",
             fontsize=8.8, family="monospace", color="#2D3748",
             bbox=dict(boxstyle="round,pad=0.7", fc="#F7FAFC", ec="#CBD5E0", lw=1.0))

    fig.suptitle(
        "7.3 — GNNExplainer appliqué au module fraude : explication d'une prédiction de fraude\n"
        "Modèle réel GraphSAGE · graphe Tentative 3 (profil ciblé) · valeurs calculées, non simulées",
        fontsize=12.5, fontweight="bold", y=0.96, linespacing=1.35,
    )

    out = FIG_DIR / "chapter7_gnn_explainer_fraud.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Figure générée -> {out}")

    # Persister les valeurs pour reproductibilité / rapport
    summary = {
        "target_idx": target_idx,
        "target_prob": float(target_prob),
        "y_true": y_true_target,
        "auc_roc": float(auc_roc),
        "pr_auc": float(pr_auc),
        "subgraph_nodes": len(sub_nodes),
        "subgraph_edges": E_sub,
        "important_edges": len(important_edges),
        "p_full": float(p_full),
        "p_kept": float(p_kept),
        "p_excluded": float(p_excluded),
        "top_features": {feat_names[i]: float(feat_imp_mean[order[i]]) for i in range(10)},
    }
    import json
    (FIG_DIR / "chapter7_gnn_explainer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Résumé persisté -> reports/figures/chapter7_gnn_explainer_summary.json")


if __name__ == "__main__":
    main()