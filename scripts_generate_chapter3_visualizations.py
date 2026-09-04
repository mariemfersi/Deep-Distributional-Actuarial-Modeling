"""
Génère les visualisations du Chapitre 3 — Données et préparation.

Sorties : reports/figures/chapter3_*.png
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.pricing.data import build_pricing_dataset
from src.pricing.features import build_features
from src.reserving.data import build_reserving_dataset

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def save_fig(name: str):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"OK {path.relative_to(PROJECT_ROOT)}")


def figure_tariff_variable_distributions(df: pd.DataFrame):
    variables = [
        ("DrivAge", "Âge du conducteur", "ans"),
        ("BonusMalus", "Coefficient bonus-malus", "coefficient"),
        ("VehPower", "Puissance du véhicule", "puissance fiscale"),
        ("VehAge", "Ancienneté du véhicule", "ans"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, (col, title, xlabel) in zip(axes, variables):
        ax.hist(df[col], bins=40, color="#2A6FBB", edgecolor="white", alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Nombre de polices")
        ax.ticklabel_format(axis="y", style="plain")

    fig.suptitle("Distributions des principales variables tarifaires — freMTPL2", fontsize=15, fontweight="bold")
    save_fig("chapter3_01_distributions_variables_tarifaires.png")


def figure_claim_frequency_by_geography(df: pd.DataFrame):
    # Fréquence actuarielle = nombre de sinistres / exposition.
    # On affiche Region et Area pour donner deux niveaux géographiques présents dans freMTPL2.
    region = df.groupby("Region").agg(claims=("ClaimNb", "sum"), exposure=("Exposure", "sum")).reset_index()
    region["frequency"] = region["claims"] / region["exposure"]
    region = region.sort_values("frequency", ascending=False)

    area = df.groupby("Area").agg(claims=("ClaimNb", "sum"), exposure=("Exposure", "sum")).reset_index()
    area["frequency"] = area["claims"] / area["exposure"]
    area = area.sort_values("frequency", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2.2, 1]})

    axes[0].bar(range(len(region)), region["frequency"], color="#4C78A8", edgecolor="white")
    axes[0].set_xticks(range(len(region)))
    axes[0].set_xticklabels(region["Region"], rotation=60, ha="right")
    axes[0].set_title("Fréquence de sinistres par région", fontweight="bold")
    axes[0].set_xlabel("Région")
    axes[0].set_ylabel("Fréquence = sinistres / exposition")

    axes[1].bar(range(len(area)), area["frequency"], color="#F58518", edgecolor="white")
    axes[1].set_xticks(range(len(area)))
    axes[1].set_xticklabels(area["Area"])
    axes[1].set_title("Fréquence par zone Area", fontweight="bold")
    axes[1].set_xlabel("Zone")
    axes[1].set_ylabel("")

    fig.suptitle("Hétérogénéité géographique de la fréquence — freMTPL2", fontsize=15, fontweight="bold")
    save_fig("chapter3_02_frequence_sinistres_geographie.png")


def plot_heatmap(ax, matrix, cmap="RdBu_r", center=None, fmt=".2f", cbar_label=""):
    data = np.asarray(matrix, dtype=float)
    masked = np.ma.masked_invalid(data)
    if center is None:
        vmin, vmax = np.nanmin(data), np.nanmax(data)
    else:
        bound = max(abs(np.nanmin(data) - center), abs(np.nanmax(data) - center))
        vmin, vmax = center - bound, center + bound
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                color = "white" if abs(data[i, j] - (center or 0)) > (vmax - vmin) * 0.30 else "black"
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=8, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    ax.grid(False)
    return im


def figure_correlation_heatmap(df: pd.DataFrame):
    cols = [
        "Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density",
        "ClaimNb", "ClaimAmount_capped",
    ]
    corr = df[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_heatmap(ax, corr.values, cmap="RdBu_r", center=0, fmt=".2f", cbar_label="Corrélation de Spearman")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    ax.set_title("Matrice de corrélation des variables explicatives et cibles", fontweight="bold")
    save_fig("chapter3_03_matrice_correlation_variables.png")


def figure_raw_development_triangle(observed: pd.DataFrame):
    tri = observed.copy()
    tri["AccidentYear_int"] = pd.to_datetime(tri["AccidentYear"]).dt.year

    # Triangle brut de portefeuille : cumul payé agrégé par année de survenance et lag.
    pivot = tri.pivot_table(
        index="AccidentYear_int",
        columns="DevelopmentLag",
        values="CumPaidLoss",
        aggfunc="sum",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(11, 7))
    plot_heatmap(ax, (pivot / 1_000_000).values, cmap="YlGnBu", fmt=".1f", cbar_label="CumPaidLoss agrégé (millions $)")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    ax.set_title("Triangle de développement brut observé — paiements cumulés", fontweight="bold")
    ax.set_xlabel("Lag de développement")
    ax.set_ylabel("Année de survenance")
    save_fig("chapter3_04_triangle_developpement_brut.png")


def add_box(ax, xy, text, width=1.8, height=0.7, color="#E8F1FB", edge="#2A6FBB", fontsize=10):
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight="bold")
    return box


def add_arrow(ax, start, end, color="#555555", rad=0.0):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="->",
        mutation_scale=13,
        linewidth=1.4,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def figure_fraud_graph_schema():
    # Important : le dataset utilisé ne contient pas réellement les nœuds assurés/tiers/réparateurs.
    # Le schéma est donc explicitement présenté comme une illustration conceptuelle de la donnée relationnelle souhaitée.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    nodes = {
        "Assuré": (1.5, 4.5, "#E8F1FB", "#2A6FBB"),
        "Police": (3.5, 4.5, "#E8F1FB", "#2A6FBB"),
        "Sinistre\n(dossier)": (5.5, 3.2, "#FFF2CC", "#B7791F"),
        "Tiers": (7.8, 4.5, "#E6F4EA", "#2F855A"),
        "Réparateur": (7.8, 2.0, "#FCE8E6", "#C53030"),
        "Expert / agent": (3.2, 1.8, "#F3E8FF", "#6B46C1"),
    }

    for label, (x, y, color, edge) in nodes.items():
        add_box(ax, (x, y), label, width=1.65, height=0.75, color=color, edge=edge, fontsize=10)

    add_arrow(ax, (2.35, 4.5), (2.65, 4.5))
    add_arrow(ax, (4.25, 4.25), (4.75, 3.55))
    add_arrow(ax, (6.35, 3.55), (7.0, 4.2))
    add_arrow(ax, (6.35, 3.05), (7.0, 2.25))
    add_arrow(ax, (4.0, 2.05), (4.75, 2.85))

    ax.text(3.0, 4.8, "détient", ha="center", fontsize=9, color="#555")
    ax.text(4.55, 4.05, "déclare", ha="center", fontsize=9, color="#555")
    ax.text(6.9, 4.05, "implique", ha="center", fontsize=9, color="#555")
    ax.text(6.95, 2.55, "répare", ha="center", fontsize=9, color="#555")
    ax.text(4.25, 2.45, "traite", ha="center", fontsize=9, color="#555")

    # Petit encadré méthodologique honnête par rapport au projet.
    note = (
        "Schéma conceptuel illustratif.\n"
        "Dans le projet, le dataset fraude ne fournit pas ces entités relationnelles complètes ;\n"
        "le graphe opérationnel est construit entre dossiers par similarité de profil."
    )
    ax.text(5, 0.55, note, ha="center", va="center", fontsize=9, color="#444",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7", edgecolor="#BBBBBB"))

    ax.set_title("Schéma relationnel cible pour la détection de fraude", fontsize=15, fontweight="bold", pad=15)
    save_fig("chapter3_05_schema_graphe_relationnel_fraude.png")


def figure_preparation_pipeline_flowchart():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    rows = [
        ("Tarification\nfreMTPL2", 5.7, [
            "CSV fréquence\n+ sévérité", "Jointure IDpol", "Nettoyage\nExposure, ClaimNb", "Features\n+ split", "GLM / CANN\nNGBoost"
        ], "#E8F1FB", "#2A6FBB"),
        ("Provisionnement\nCAS ppauto", 3.5, [
            "Triangle complet", "Cumul →\nincrémental", "Split observé\n/ futur", "Filtre prime\nminimale", "Mack /\nDeep Triangle"
        ], "#E6F4EA", "#2F855A"),
        ("Fraude\nfraud_oracle", 1.3, [
            "Dossiers\nsinistres", "Encodage\nvariables", "Labels fraude", "Arêtes par\nsimilarité", "RF / GNN\nexplicabilité"
        ], "#FFF2CC", "#B7791F"),
    ]

    for label, y, steps, color, edge in rows:
        add_box(ax, (1.1, y), label, width=1.8, height=0.8, color=color, edge=edge, fontsize=9)
        x_positions = [3.0, 5.2, 7.4, 9.6, 11.8]
        for x, step in zip(x_positions, steps):
            add_box(ax, (x, y), step, width=1.65, height=0.8, color="white", edge=edge, fontsize=8.5)
        add_arrow(ax, (2.0, y), (2.2, y), color=edge)
        for x1, x2 in zip(x_positions[:-1], x_positions[1:]):
            add_arrow(ax, (x1 + 0.85, y), (x2 - 0.85, y), color=edge)

    ax.text(12.9, 3.5, "Données\npréparées\npour modélisation", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7", edgecolor="#777777"))
    add_arrow(ax, (12.65, 5.7), (12.65, 4.25), color="#777")
    add_arrow(ax, (12.65, 3.5), (12.15, 3.5), color="#777")
    add_arrow(ax, (12.65, 1.3), (12.65, 2.75), color="#777")

    ax.set_title("Pipeline de préparation des trois sources de données", fontsize=15, fontweight="bold", pad=15)
    save_fig("chapter3_06_pipeline_preparation_donnees.png")


def main():
    print("Chargement des données pricing...")
    pricing_df = build_pricing_dataset()
    pricing_df = build_features(pricing_df)

    print("Chargement des données reserving...")
    _, observed, _ = build_reserving_dataset()

    print("Génération des figures Chapitre 3...")
    figure_tariff_variable_distributions(pricing_df)
    figure_claim_frequency_by_geography(pricing_df)
    figure_correlation_heatmap(pricing_df)
    figure_raw_development_triangle(observed)
    figure_fraud_graph_schema()
    figure_preparation_pipeline_flowchart()
    print("Terminé.")


if __name__ == "__main__":
    main()
