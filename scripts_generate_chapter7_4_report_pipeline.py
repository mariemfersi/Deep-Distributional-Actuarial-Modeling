"""Figure 7.4 — Vers un rapport actuariel automatisé : pipeline d'analyse intégrée."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Palette ──────────────────────────────────────────────────────
COL_INPUT   = "#2196F3"   # bleu — données
COL_PRICING = "#4CAF50"   # vert — tarification
COL_FRAUD   = "#F44336"   # rouge — fraude
COL_RESERV  = "#FF9800"   # orange — provisions
COL_EXPLAIN = "#9C27B0"   # violet — explicabilité
COL_REPORT  = "#607D8B"   # gris-bleu — rapport
COL_LLM     = "#795548"   # marron — LLM/perspective
COL_BG      = "#FAFAFA"
COL_ARROW   = "#37474F"
COL_LIGHT   = {COL_INPUT: "#E3F2FD", COL_PRICING: "#E8F5E9", COL_FRAUD: "#FFEBEE",
               COL_RESERV: "#FFF3E0", COL_EXPLAIN: "#F3E5F5", COL_REPORT: "#ECEFF1",
               COL_LLM: "#EFEBE9"}

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── Titre ────────────────────────────────────────────────────────
ax.text(9, 9.6, "7.4 — Vers un rapport actuariel automatisé",
        fontsize=16, fontweight="bold", ha="center", va="center",
        fontfamily="sans-serif")
ax.text(9, 9.25, "Pipeline d'analyse intégrée tarification · fraude · provisions · explicabilité",
        fontsize=10, ha="center", va="center", color="#555", fontfamily="sans-serif")


def draw_box(x, y, w, h, color, label, sublabel=None, fontsize=9):
    """Dessine un boîtier arrondi avec ombre."""
    light = COL_LIGHT.get(color, "#f5f5f5")
    # Ombre
    shadow = FancyBboxPatch((x + 0.04, y - 0.04), w, h,
                            boxstyle="round,pad=0.08", facecolor="#ddd",
                            edgecolor="none", alpha=0.3, zorder=1)
    ax.add_patch(shadow)
    # Boîte
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.08", facecolor=light,
                         edgecolor=color, linewidth=1.8, zorder=2)
    ax.add_patch(box)
    # Texte
    cy = y + h / 2
    if sublabel:
        ax.text(x + w / 2, cy + 0.15, label, fontsize=fontsize,
                fontweight="bold", ha="center", va="center", color=color, zorder=3)
        ax.text(x + w / 2, cy - 0.2, sublabel, fontsize=7,
                ha="center", va="center", color="#666", zorder=3,
                fontfamily="sans-serif")
    else:
        ax.text(x + w / 2, cy, label, fontsize=fontsize,
                fontweight="bold", ha="center", va="center", color=color, zorder=3)


def draw_arrow(x1, y1, x2, y2, color=COL_ARROW, lw=1.5, style="-|>"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color,
                            linewidth=lw, mutation_scale=12,
                            connectionstyle="arc3,rad=0",
                            zorder=4)
    ax.add_patch(arrow)


def draw_arrow_curved(x1, y1, x2, y2, color=COL_ARROW, lw=1.5, rad=0.15):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>", color=color,
                            linewidth=lw, mutation_scale=12,
                            connectionstyle=f"arc3,rad={rad}",
                            zorder=4)
    ax.add_patch(arrow)


# ── Ligne 1 : Sources de données ─────────────────────────────────
# Boîte Données
draw_box(0.3, 7.5, 2.6, 1.2, COL_INPUT, "Données", "freMTPL2 · CAS Reserve\nPostgreSQL")
draw_box(0.3, 5.8, 2.6, 1.2, COL_INPUT, "Graphe Fraude", "Tabular → GNN\nTentative 3 (5 attr)")

# ── Ligne 2 : Modules de modélisation ────────────────────────────
draw_box(4.0, 7.5, 2.8, 1.2, COL_PRICING, "Tarification", "GLM Poisson/Gamma\nNGBoost · CANN · Copule")
draw_box(4.0, 5.8, 2.8, 1.2, COL_FRAUD, "Détection Fraude", "Random Forest\nGraphSAGE (GNN)")
draw_box(4.0, 4.1, 2.8, 1.2, COL_RESERV, "Provisions", "Mack Chain-Ladder\nDeep Triangle GRU")

# ── Ligne 3 : Explicabilité ──────────────────────────────────────
draw_box(8.0, 7.5, 2.6, 1.2, COL_EXPLAIN, "SHAP Values", "GLM: LinearExplainer\nNGBoost: KernelExplainer")
draw_box(8.0, 5.8, 2.6, 1.2, COL_EXPLAIN, "Interactions SHAP", "Effets croisés\nCovariance features")
draw_box(8.0, 4.1, 2.6, 1.2, COL_EXPLAIN, "GNNExplainer", "Sous-graphe causalement\nlié à la prédiction")

# ── Ligne 4 : Agrégation et rapport ──────────────────────────────
draw_box(11.8, 5.8, 3.2, 2.9, COL_REPORT, "Agrégation\nMulti-modèle",
         "Tableaux de bord\nMétriques · Comparaisons\nSeuils · Alertes")

draw_box(15.5, 7.2, 2.2, 1.5, COL_LLM, "LLM Module",
         "Génération\n automatique\n de texte")

draw_box(15.5, 5.2, 2.2, 1.5, COL_REPORT, "Rapport\nActuariel",
         "PDF · HTML\nNarratif + chiffres")

# ── Flèches : données → modèles ──────────────────────────────────
draw_arrow(2.9, 8.1, 4.0, 8.1)  # Données → Tarification
draw_arrow(2.9, 6.4, 4.0, 6.4)  # Graphe → Fraude
draw_arrow(1.6, 7.5, 1.6, 7.0)  # Données → Graphe
draw_arrow(1.6, 5.8, 1.6, 5.3)  # Données → Provisions
draw_arrow(2.9, 5.3, 4.0, 4.7)  # Données → Provisions

# ── Flèches : modèles → explicabilité ────────────────────────────
draw_arrow(6.8, 8.1, 8.0, 8.1)  # Tarification → SHAP
draw_arrow(6.8, 6.4, 8.0, 6.4)  # Fraude → Interactions
draw_arrow(6.8, 4.7, 8.0, 4.7)  # Provisions → GNNExplainer

# ── Flèches : explicabilité → agrégation ─────────────────────────
draw_arrow(10.6, 8.1, 11.8, 7.5)   # SHAP → Agrégation
draw_arrow(10.6, 6.4, 11.8, 6.8)   # Interactions → Agrégation
draw_arrow(10.6, 4.7, 11.8, 6.2)   # GNNExplainer → Agrégation

# ── Flèches : agrégation → rapport ──────────────────────────────
draw_arrow(15.0, 7.2, 15.5, 7.9)   # Agrégation → LLM
draw_arrow(15.0, 6.5, 15.5, 5.9)   # Agrégation → Rapport
draw_arrow(16.6, 7.2, 16.6, 6.7)   # LLM → Rapport

# ── Flèches latérales : feedback ────────────────────────────────
draw_arrow_curved(11.8, 5.8, 6.8, 4.5, color="#90A4AE", lw=1.0, rad=-0.2)
draw_arrow_curved(11.8, 5.8, 6.8, 6.2, color="#90A4AE", lw=1.0, rad=-0.15)
draw_arrow_curved(11.8, 5.8, 6.8, 7.9, color="#90A4AE", lw=1.0, rad=-0.25)
ax.text(8.5, 3.4, "Boucle de rétroaction :\nalertes → recalibration",
        fontsize=7, ha="center", va="center", color="#90A4AE", style="italic",
        fontfamily="sans-serif")

# ── Encadré "Perspective" en bas ─────────────────────────────────
persp_box = FancyBboxPatch((0.3, 0.4), 17.4, 1.6,
                           boxstyle="round,pad=0.1",
                           facecolor="#FFF8E1", edgecolor="#F9A825",
                           linewidth=1.5, linestyle="--", zorder=2)
ax.add_patch(persp_box)
ax.text(9, 1.65, "Perspective — Vers un rapport actuariel automatisé",
        fontsize=11, fontweight="bold", ha="center", va="center",
        color="#F57F17", fontfamily="sans-serif")
ax.text(9, 1.1,
        "Intégration d'un LLM (ex : Claude, GPT) pour générer automatiquement le narratif actuariel à partir des résultats SHAP,\n"
        "des métriques de modèle et des tableaux de bord. Le modèle fournit les chiffres ; le LLM rédige l'interprétation.\n"
        "Sécurité : validation humaine obligatoire avant diffusion — le LLM est un assistante, pas un décideur.",
        fontsize=8.5, ha="center", va="center", color="#555",
        fontfamily="sans-serif", linespacing=1.5)

# ── Légende ──────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=COL_LIGHT[COL_INPUT], edgecolor=COL_INPUT, label="Données d'entrée"),
    mpatches.Patch(facecolor=COL_LIGHT[COL_PRICING], edgecolor=COL_PRICING, label="Tarification"),
    mpatches.Patch(facecolor=COL_LIGHT[COL_FRAUD], edgecolor=COL_FRAUD, label="Détection fraude"),
    mpatches.Patch(facecolor=COL_LIGHT[COL_RESERV], edgecolor=COL_RESERV, label="Provisions"),
    mpatches.Patch(facecolor=COL_LIGHT[COL_EXPLAIN], edgecolor=COL_EXPLAIN, label="Explicabilité"),
    mpatches.Patch(facecolor=COL_LIGHT[COL_REPORT], edgecolor=COL_REPORT, label="Reporting"),
    mpatches.Patch(facecolor=COL_LIGHT[COL_LLM], edgecolor=COL_LLM, label="LLM (perspective)"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=7.5,
          framealpha=0.9, edgecolor="#ccc", ncol=4,
          bbox_to_anchor=(0.0, -0.02))

plt.tight_layout(pad=0.5)
out = "reports/figures/chapter7_automated_actuarial_report_pipeline.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Figure 7.4 sauvegardee -> {out}")
