"""
Module Fraude — Construction du graphe de dossiers.

Le graphe combine une ancre réelle du dataset (RepNumber, l'agent traitant
le dossier) et une condition de similarité de profil (Make, VehicleCategory,
PolicyType), documentée comme limite méthodologique au chapitre 3 : ce
dataset ne fournit pas de structure relationnelle complète et vérifiée,
mais RepNumber constitue une vraie variable d'ancrage plutôt qu'une
simulation entièrement artificielle.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def build_edge_index(df: pd.DataFrame, min_shared_attrs: int = 2) -> torch.Tensor:
    """
    Construit les arêtes du graphe : deux dossiers sont connectés s'ils
    partagent le même RepNumber ET au moins `min_shared_attrs` autres
    attributs de profil (Make, VehicleCategory, PolicyType).
    """
    edges = []
    similarity_cols = ["Make", "VehicleCategory", "PolicyType"]

    for rep, group in df.groupby("RepNumber"):
        if len(group) < 2:
            continue
        indices = group.index.tolist()
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]
                shared = sum(
                    df.loc[idx_i, col] == df.loc[idx_j, col] for col in similarity_cols
                )
                if shared >= min_shared_attrs:
                    edges.append((idx_i, idx_j))
                    edges.append((idx_j, idx_i))  # graphe non orienté

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def build_pyg_graph(df: pd.DataFrame, feature_cols: list, min_shared_attrs: int = 2) -> Data:
    """Construit l'objet Data PyTorch Geometric complet (features + arêtes + labels)."""
    df = df.reset_index(drop=True)

    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df["fraud_label"].values, dtype=torch.long)

    edge_index = build_edge_index(df, min_shared_attrs=min_shared_attrs)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data



def build_edge_index(df: pd.DataFrame, similarity_cols: list, min_shared_attrs: int) -> torch.Tensor:
    """
    Construit les arêtes du graphe par similarité de profil stricte.
    Faute d'identifiant relationnel fiable dans ce dataset (RepNumber
    s'avère être un code régional à faible cardinalité, pas un agent
    individuel), le graphe est construit entièrement par similarité,
    limite méthodologique documentée explicitement au chapitre 3.
    """
    from collections import defaultdict

    edges = []
    # Regrouper par tuple de valeurs sur les colonnes de similarité -- bien plus efficace
    # que la double boucle précédente, et équivalent à "min_shared_attrs = tous les attributs"
    groups = defaultdict(list)
    for idx, row in df[similarity_cols].iterrows():
        key = tuple(row.values)
        groups[key].append(idx)

    for key, indices in groups.items():
        if len(indices) < 2 or len(indices) > 50:  # on ignore aussi les groupes trop massifs
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                edges.append((indices[i], indices[j]))
                edges.append((indices[j], indices[i]))

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_edge_index_rare_shared(df: pd.DataFrame, cols: list, rarity_threshold: float = 0.10) -> torch.Tensor:
    """
    Connecte deux dossiers s'ils partagent la MÊME valeur RARE (fréquence
    < rarity_threshold) sur au moins une des colonnes fournies. Cible
    spécifiquement les combinaisons atypiques plutôt que le profil général,
    contrairement aux tentatives précédentes qui diluaient le signal en
    connectant sur des valeurs fréquentes et peu informatives.
    """
    from collections import defaultdict

    edges = set()

    for col in cols:
        freq = df[col].value_counts(normalize=True)
        rare_values = freq[freq < rarity_threshold].index

        for val in rare_values:
            indices = df.index[df[col] == val].tolist()
            if len(indices) < 2 or len(indices) > 200:
                continue
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    edges.add((indices[i], indices[j]))
                    edges.add((indices[j], indices[i]))

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edge_list = list(edges)
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()