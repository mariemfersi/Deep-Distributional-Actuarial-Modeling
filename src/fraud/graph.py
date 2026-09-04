"""
Module Fraude — Construction du graphe de dossiers.

Fournit 4 stratégies de construction de graphes par similarité ou ancrage,
chacune évaluée par un test d'homophilie dans le chapitre 6.
"""

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch_geometric.data import Data


def build_edge_index_repnumber(df: pd.DataFrame, min_shared_attrs: int = 2) -> torch.Tensor:
    """
    Tentative 1 : Deux dossiers sont connectés s'ils partagent le même RepNumber
    ET au moins `min_shared_attrs` autres attributs (Make, VehicleCategory, PolicyType).
    """
    edges = []
    similarity_cols = ["Make", "VehicleCategory", "PolicyType"]

    for rep, group in df.groupby("RepNumber"):
        if len(group) < 2:
            continue
        indices = group.index.tolist()
        vals = group[similarity_cols].values
        n = len(indices)
        for i in range(n):
            for j in range(i + 1, n):
                shared = (vals[i] == vals[j]).sum()
                if shared >= min_shared_attrs:
                    edges.append((indices[i], indices[j]))
                    edges.append((indices[j], indices[i]))

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_edge_index_similarity(df: pd.DataFrame, similarity_cols: list, min_shared_attrs: int = None) -> torch.Tensor:
    """
    Tentatives 2 et 3 : Connexion par correspondance exacte sur une liste d'attributs.
    - Tentative 2 : 6 attributs génériques (Make, VehicleCategory, PolicyType, AccidentArea, AgeOfVehicle, BasePolicy)
    - Tentative 3 : 5 attributs ciblés discriminants (Fault, AddressChange_Claim, Days_Policy_Claim, PolicyType, BasePolicy)
    """
    edges = []
    groups = defaultdict(list)
    for idx, row in df[similarity_cols].iterrows():
        key = tuple(row.values)
        groups[key].append(idx)

    for key, indices in groups.items():
        if len(indices) < 2 or len(indices) > 50:
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
    Tentative 4 : Connecte deux dossiers s'ils partagent la MÊME valeur RARE
    (fréquence < rarity_threshold) sur au moins une des colonnes fournies.
    """
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


def build_pyg_graph(df: pd.DataFrame, feature_cols: list, strategy: str = "repnumber", **kwargs) -> Data:
    """Construit l'objet Data PyTorch Geometric selon la stratégie choisie."""
    df = df.reset_index(drop=True)

    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df["fraud_label"].values, dtype=torch.long)

    if strategy == "repnumber":
        edge_index = build_edge_index_repnumber(df, **kwargs)
    elif strategy == "similarity":
        edge_index = build_edge_index_similarity(df, **kwargs)
    elif strategy == "rare":
        edge_index = build_edge_index_rare_shared(df, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    data = Data(x=x, edge_index=edge_index, y=y)
    return data