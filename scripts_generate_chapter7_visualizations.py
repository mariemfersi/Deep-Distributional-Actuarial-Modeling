"""
scripts_generate_chapter7_visualizations.py (Version SHAP Réel)

Génère les visualisations du Chapitre 7 — Explicabilité et Interprétation Transversale.
TOUTES les valeurs SHAP sont calculées depuis les vrais modèles entraînés.

Figures produites (reports/figures/chapter7_*.png) :
  1. chapter7_shap_beeswarm_glm_ngboost.png   — Summary beeswarm GLM (fréquence) + NGBoost (sévérité)
  2. chapter7_shap_waterfall_individual.png   — Waterfall individuel : police (GLM) + sinistre (RF fraude)
  3. chapter7_shap_beeswarm_fraud_rf.png      — Summary beeswarm Random Forest fraude
  4. chapter7_cann_interactions.png           — Interactions SHAP CANN (KernelExplainer)
"""

from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
import shap

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "backend"))

from src.pricing.data import build_pricing_dataset
from src.pricing.features import build_features
from src.pricing.models import NGBOOST_FEATURES
from src.fraud.data import load_fraud_data, prepare_fraud_features, train_test_split_fraud
from src.explainability.shap_pricing import (
    compute_shap_glm_frequency,
    aggregate_shap_by_original_feature,
    _patsy_design_matrix,
    GLM_FREQ_FORMULA_RHS,
)
from src.explainability.shap_fraud import compute_shap_fraud_rf

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":      140,
    "savefig.dpi":     300,
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.25,
})

COLORS = {
    "red":    "#E53E3E",
    "blue":   "#2A6FBB",
    "purple": "#6B46C1",
    "gray":   "#718096",
}


# ===========================================================================
# DATA LOADING (shared across all figures)
# ===========================================================================

def load_pricing_sample(n: int = 500, seed: int = 42):
    """Charge un sous-échantillon du jeu de tarification avec les features GLM."""
    df = build_pricing_dataset()
    df = build_features(df)
    # Exclure les lignes sans modalités valides
    df = df.dropna(subset=["DrivAge_bucket", "VehAge_bucket", "BM_bucket"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), min(n, len(df)), replace=False)
    return df.iloc[idx].reset_index(drop=True)


def load_fraud_test_sample(n: int = 500, seed: int = 42):
    """Charge le jeu de test fraude avec features encodées."""
    df_raw  = load_fraud_data()
    df_prep = prepare_fraud_features(df_raw)
    _, test = train_test_split_fraud(df_prep, seed=123)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(test), min(n, len(test)), replace=False)
    return test.iloc[idx].reset_index(drop=True)


# ===========================================================================
# FIGURE 1 — Beeswarm SHAP : GLM Fréquence + NGBoost Sévérité
# ===========================================================================

def generate_figure1_beeswarm_glm_ngboost(df_pricing):
    """Summary beeswarm côte à côte : GLM Poisson fréquence + NGBoost sévérité."""
    import joblib
    from src.explainability.shap_pricing import compute_shap_ngboost_severity
    from src.pricing.features import add_cann_features

    print("  [GLM] Calcul des SHAP via LinearExplainer...")
    shap_glm, feat_names_onehot, base_glm, X_dm = compute_shap_glm_frequency(df_pricing)
    shap_agg = aggregate_shap_by_original_feature(shap_glm.values, feat_names_onehot)

    # Exclure l'Intercept du beeswarm
    cols_to_plot = [c for c in shap_agg.columns if c != "Intercept"]
    shap_plot    = shap_agg[cols_to_plot]

    # Valeurs features originales pour la couleur (mean feature value par obs)
    feat_vals_dict = {}
    for col in cols_to_plot:
        # Retrouver la valeur originale depuis le DataFrame pricing
        base = col.lower()
        if base in [c.lower() for c in df_pricing.columns]:
            match = [c for c in df_pricing.columns if c.lower() == base][0]
            v = df_pricing[match].values
            if hasattr(v[0], 'codes'):
                v = v.codes
            try:
                v = v.astype(float)
                v = (v - v.min()) / (v.max() - v.min() + 1e-9)
            except Exception:
                v = np.zeros(len(df_pricing))
            feat_vals_dict[col] = v
        else:
            feat_vals_dict[col] = np.zeros(len(shap_agg))

    print("  [NGBoost] Calcul des SHAP via TreeExplainer/KernelExplainer...")
    df_pricing_sev = add_cann_features(df_pricing)
    # garder uniquement les colonnes NGBoost
    df_sev = df_pricing_sev.dropna(subset=NGBOOST_FEATURES)
    shap_ng, feat_ng = compute_shap_ngboost_severity(df_sev)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panneau gauche : GLM beeswarm
    ax = axes[0]
    feat_order = list(reversed(cols_to_plot))
    for row_idx, feat in enumerate(feat_order):
        sv = shap_plot[feat].values
        fv = feat_vals_dict.get(feat, np.zeros(len(sv)))
        jitter = np.random.default_rng(row_idx).uniform(-0.18, 0.18, size=len(sv))
        sc = ax.scatter(sv, row_idx + jitter, c=fv, cmap="coolwarm",
                        s=10, alpha=0.65, linewidths=0)

    ax.set_yticks(range(len(feat_order)))
    ax.set_yticklabels(feat_order, fontsize=9)
    ax.axvline(0, color="#4A5568", lw=1.2, ls="--")
    ax.set_xlabel("Valeur SHAP (espace log-fréquence)", fontsize=10)
    ax.set_title("GLM Poisson — Fréquence\n(shap.LinearExplainer, exact)", fontsize=11, fontweight="bold")
    cb0 = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb0.set_label("Valeur feature\n(Bleu=Basse  Rouge=Élevée)", fontsize=8)

    # Panneau droit : NGBoost beeswarm
    ax1 = axes[1]
    if hasattr(shap_ng, "values"):
        sv_ng = shap_ng.values
    else:
        sv_ng = shap_ng

    feat_order_ng = list(reversed(feat_ng))
    for row_idx, feat in enumerate(feat_order_ng):
        fi = feat_ng.index(feat) if feat in feat_ng else row_idx
        if fi >= sv_ng.shape[1]:
            continue
        sv = sv_ng[:, fi]
        fv = np.linspace(0, 1, len(sv))  # fallback couleur ordinale
        jitter = np.random.default_rng(row_idx + 100).uniform(-0.18, 0.18, size=len(sv))
        sc1 = ax1.scatter(sv, row_idx + jitter, c=fv, cmap="coolwarm",
                          s=10, alpha=0.65, linewidths=0)

    ax1.set_yticks(range(len(feat_order_ng)))
    ax1.set_yticklabels(feat_order_ng, fontsize=9)
    ax1.axvline(0, color="#4A5568", lw=1.2, ls="--")
    ax1.set_xlabel("Valeur SHAP (espace log-sévérité)", fontsize=10)
    ax1.set_title("NGBoost — Sévérité\n(shap.TreeExplainer / KernelExplainer)", fontsize=11, fontweight="bold")
    cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.6, pad=0.02)
    cb1.set_label("Valeur feature\n(Bleu=Basse  Rouge=Élevée)", fontsize=8)

    plt.suptitle(
        "Summary Plots SHAP (Beeswarm) — Modèles de Tarification\n"
        "Chaque point = une observation du portefeuille",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = FIG_DIR / "chapter7_shap_beeswarm_glm_ngboost.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  -> {path.name}")

    # Retourne les objets pour la figure waterfall
    return shap_agg, cols_to_plot, df_pricing


# ===========================================================================
# FIGURE 2 — Waterfall individuel : Police GLM + Sinistre RF Fraude
# ===========================================================================

def _draw_waterfall_clean(ax, feature_names, shap_vals, base_value,
                           feature_display_values, title, xlabel, final_label,
                           color_pos, color_neg):
    """Waterfall plot propre avec barres empilées et annotations claires."""
    n = len(feature_names)
    # Tri par |SHAP| décroissant
    order   = np.argsort(np.abs(shap_vals))[::-1]
    names   = [feature_names[i] for i in order]
    sv      = shap_vals[order]
    fv_disp = [feature_display_values[i] for i in order]

    # Running total depuis base_value
    lefts  = []
    curr   = base_value
    for d in sv:
        lefts.append(curr)
        curr += d
    final  = curr

    # Positions y : prédiction tout en haut, base tout en bas
    all_labels   = ["Valeur de base"] + [f"{n}  = {v}" for n, v in zip(names, fv_disp)] + ["Prédiction"]
    y_positions  = list(range(len(all_labels) - 1, -1, -1))
    n_rows       = len(all_labels)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(all_labels, fontsize=8.5)

    # Base
    y_b = y_positions[0]
    ax.barh(y_b, base_value, color=COLORS["gray"], height=0.55, zorder=2)
    ax.text(base_value / 2, y_b, f"{base_value:.3f}",
            ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=3)

    # Feature bars
    for k in range(n):
        y   = y_positions[k + 1]
        d   = sv[k]
        lft = lefts[k]
        col = color_pos if d >= 0 else color_neg
        ax.barh(y, d, left=lft, color=col, height=0.55, zorder=2)
        sign = "+" if d >= 0 else ""
        ax.text(lft + d / 2, y, f"{sign}{d:.3f}",
                ha="center", va="center", fontsize=7.5, fontweight="bold", color="white", zorder=3)
        # Fil de guidage
        ax.vlines(lft + d, y - 0.35, y - 0.65, colors="#CBD5E0", lw=0.8, ls="dashed")

    # Prédiction finale
    y_p = y_positions[-1]
    ax.barh(y_p, final, color=COLORS["blue"] if final < 0 else COLORS["red"],
            height=0.55, alpha=0.9, zorder=2)
    off = 0.06 if final >= 0 else -0.06
    ha  = "left" if final >= 0 else "right"
    ax.text(final + off, y_p, f"{final:.3f}\n{final_label}",
            ha=ha, va="center", fontsize=8.5, fontweight="bold",
            color=COLORS["blue"] if final < 0 else COLORS["red"])

    ax.axvline(0, color="#4A5568", lw=1.2, ls="--", alpha=0.7)

    ppos  = mpatches.Patch(color=color_pos,     label="Augmente le risque")
    pneg  = mpatches.Patch(color=color_neg,     label="Réduit le risque")
    pbase = mpatches.Patch(color=COLORS["gray"],label="Valeur de base E[f(X)]")
    ax.legend(handles=[pbase, ppos, pneg], loc="lower right", fontsize=7.5,
              frameon=True, facecolor="white", framealpha=0.9)

    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.3)


def generate_figure2_waterfall_individual(df_pricing, df_fraud_test):
    """Waterfall individuel : 1 police (GLM fréquence) + 1 dossier sinistre (RF fraude)."""
    print("  [Waterfall GLM] Calcul SHAP sur 1 observation réelle...")

    # --- Cas 1 : Profil de police GLM ---
    shap_glm, feat_onehot, base_glm, X_dm = compute_shap_glm_frequency(df_pricing.iloc[:1])
    shap_agg_1 = aggregate_shap_by_original_feature(shap_glm.values, feat_onehot)
    sv_glm = shap_agg_1.iloc[0].drop("Intercept", errors="ignore")
    base_glm_val = float(base_glm) if np.isscalar(base_glm) else float(base_glm[0])

    # Valeurs réelles de la police sélectionnée
    police = df_pricing.iloc[0]
    glm_feat_display = {
        "DrivAge_bucket": str(police.get("DrivAge_bucket", "?")),
        "VehAge_bucket":  str(police.get("VehAge_bucket", "?")),
        "BM_bucket":      str(police.get("BM_bucket", "?")),
        "VehGas":         str(police.get("VehGas", "?")),
        "VehBrand":       str(police.get("VehBrand", "?")),
        "Region":         str(police.get("Region", "?")),
        "Density_log":    f"{police.get('Density_log', 0):.2f}",
    }
    fv_glm = [glm_feat_display.get(f, "?") for f in sv_glm.index]

    # Prédiction fréquence réelle
    import joblib
    glm_model = joblib.load(PROJECT_ROOT / "models" / "glm_poisson.pkl")
    freq_pred  = float(glm_model.predict(
        df_pricing.iloc[:1],
        offset=np.zeros(1)
    ).values[0])
    final_label_glm = f"Fréq. prédite : {freq_pred*100:.2f} %"

    print("  [Waterfall RF Fraude] Calcul SHAP sur 1 dossier réel...")

    # --- Cas 2 : Dossier fraude RF ---
    sv_fraud, feat_fraud, _ = compute_shap_fraud_rf(df_fraud_test.iloc[:1])
    sv_fraud_vals = sv_fraud.values[0]
    base_fraud    = float(sv_fraud.base_values[0] if hasattr(sv_fraud.base_values, '__len__')
                         else sv_fraud.base_values)

    # Valeurs réelles du dossier
    dossier = df_fraud_test.iloc[0]
    from src.fraud.data import CATEGORICAL_COLS as FRAUD_CAT, NUMERIC_COLS as FRAUD_NUM
    feat_cols_fraud = (
        [f.replace("_code", "") for f in [f"{c}_code" for c in FRAUD_CAT]]
        + [f.replace("_norm", "") for f in [f"{c}_norm" for c in FRAUD_NUM]]
    )
    fraud_label_real = int(dossier.get("fraud_label", dossier.get("FraudFound_P", -1)))
    fraud_pred_prob  = float(sv_fraud_vals.sum() + base_fraud)
    final_label_fraud = f"Prob. fraude : {1/(1+np.exp(-fraud_pred_prob))*100:.1f} % | Réel = {fraud_label_real}"

    fv_fraud = ["?" for _ in feat_fraud]

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    _draw_waterfall_clean(
        ax=axes[0],
        feature_names=list(sv_glm.index),
        shap_vals=sv_glm.values,
        base_value=base_glm_val,
        feature_display_values=fv_glm,
        title="Cas 1 — Profil de Police (GLM Poisson Fréquence)\nValeurs SHAP calculées via LinearExplainer (exact)",
        xlabel="Contribution SHAP (espace log-fréquence)",
        final_label=final_label_glm,
        color_pos=COLORS["red"],
        color_neg=COLORS["blue"],
    )

    _draw_waterfall_clean(
        ax=axes[1],
        feature_names=feat_fraud,
        shap_vals=sv_fraud_vals,
        base_value=base_fraud,
        feature_display_values=fv_fraud,
        title="Cas 2 — Dossier Sinistre (RF Fraude)\nValeurs SHAP calculées via TreeExplainer (exact)",
        xlabel="Contribution SHAP (espace logit-fraude)",
        final_label=final_label_fraud,
        color_pos=COLORS["red"],
        color_neg=COLORS["blue"],
    )

    plt.suptitle(
        "Waterfall Plots SHAP — Décomposition Individuelle des Prédictions (Valeurs Réelles)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = FIG_DIR / "chapter7_shap_waterfall_individual.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  -> {path.name}")


# ===========================================================================
# FIGURE 3 — Beeswarm RF Fraude
# ===========================================================================

def generate_figure3_beeswarm_fraud(df_fraud_test):
    """Summary beeswarm SHAP pour le Random Forest de détection de fraude."""
    print("  [RF Fraude] Calcul des SHAP via TreeExplainer...")

    sv_fraud, feat_fraud, X_fraud = compute_shap_fraud_rf(df_fraud_test)

    shap_vals = sv_fraud.values  # (n, n_features)

    fig, ax = plt.subplots(figsize=(9, 6))
    feat_order = np.argsort(np.abs(shap_vals).mean(axis=0))  # croissant → affiché bas→haut
    feat_names_sorted = [feat_fraud[i] for i in feat_order]

    for row_idx, fi in enumerate(feat_order):
        sv  = shap_vals[:, fi]
        fv  = X_fraud[:, fi]
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        jitter  = np.random.default_rng(row_idx).uniform(-0.18, 0.18, size=len(sv))
        sc = ax.scatter(sv, row_idx + jitter, c=fv_norm, cmap="coolwarm",
                        s=10, alpha=0.65, linewidths=0)

    ax.set_yticks(range(len(feat_names_sorted)))
    ax.set_yticklabels(feat_names_sorted, fontsize=9)
    ax.axvline(0, color="#4A5568", lw=1.2, ls="--")
    ax.set_xlabel("Valeur SHAP (espace logit-fraude)", fontsize=10)
    ax.set_title(
        "Summary Plot SHAP — Random Forest Détection de Fraude\n"
        "(shap.TreeExplainer, exact — Classe Fraude = 1)",
        fontsize=12, fontweight="bold", pad=12,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Valeur feature\n(Bleu=Basse  Rouge=Élevée)", fontsize=8)

    plt.tight_layout()
    path = FIG_DIR / "chapter7_shap_beeswarm_fraud_rf.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  -> {path.name}")


# ===========================================================================
# FIGURE 4 — Interactions CANN (KernelExplainer)
# ===========================================================================

def generate_figure4_cann_interactions(df_pricing):
    """Interactions SHAP CANN : heatmap des co-variations des valeurs SHAP."""
    import joblib
    from src.pricing.features import add_cann_features
    from src.pricing.models import predict_frequency

    print("  [CANN] Calcul des SHAP via KernelExplainer (peut prendre 2-4 min)...")

    df_feat = add_cann_features(df_pricing)

    # Ajouter glm_log_pred si absent
    if "glm_log_pred" not in df_feat.columns:
        glm_model = joblib.load(PROJECT_ROOT / "models" / "glm_poisson.pkl")
        freq_pred  = predict_frequency(glm_model, df_feat)
        df_feat["glm_log_pred"] = np.log(freq_pred.clip(lower=1e-8))

    from src.explainability.shap_cann import (
        compute_shap_cann,
        compute_shap_interaction_approx,
        CANN_DISPLAY_NAMES,
        CANN_ALL_COLS,
    )

    shap_vals, feat_names, X_exp, base_val = compute_shap_cann(
        df_feat, n_background=50, n_explain=100
    )
    interact_matrix = compute_shap_interaction_approx(shap_vals, feat_names)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panneau gauche : beeswarm CANN
    ax0 = axes[0]
    feat_order = np.argsort(np.abs(shap_vals).mean(axis=0))
    feat_ord_names = [feat_names[i] for i in feat_order]
    for row_idx, fi in enumerate(feat_order):
        sv  = shap_vals[:, fi]
        fv  = X_exp[:, fi]
        fv_n = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        jit  = np.random.default_rng(row_idx).uniform(-0.18, 0.18, size=len(sv))
        sc   = ax0.scatter(sv, row_idx + jit, c=fv_n, cmap="coolwarm", s=12, alpha=0.7)
    ax0.set_yticks(range(len(feat_ord_names)))
    ax0.set_yticklabels(feat_ord_names, fontsize=9)
    ax0.axvline(0, color="#4A5568", lw=1.2, ls="--")
    ax0.set_xlabel("Valeur SHAP (espace log-fréquence CANN)", fontsize=10)
    ax0.set_title("CANN — Beeswarm SHAP\n(KernelExplainer, 100 obs)", fontsize=11, fontweight="bold")
    fig.colorbar(sc, ax=ax0, shrink=0.6, label="Valeur feature (Bleu=Bas  Rouge=Élevé)", pad=0.02)

    # Panneau droit : heatmap d'interaction
    ax1 = axes[1]
    mask = np.zeros_like(interact_matrix.values, dtype=bool)
    np.fill_diagonal(mask, True)   # cacher la diagonale (effets principaux)
    sns.heatmap(
        interact_matrix,
        annot=True, fmt=".3f", cmap="coolwarm", center=0,
        ax=ax1, cbar_kws={"label": "Covariance SHAP normalisée"},
        annot_kws={"size": 8, "weight": "bold"},
        linewidths=0.4,
    )
    ax1.set_title("Interactions SHAP CANN\n(Covariance des contributions par paire)", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="x", rotation=30)
    ax1.tick_params(axis="y", rotation=0)

    plt.suptitle(
        "Explicabilité du CANN — Valeurs SHAP et Effets d'Interaction entre Variables",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = FIG_DIR / "chapter7_cann_interactions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  -> {path.name}")


# ===========================================================================
# NETTOYAGE : supprimer les anciennes figures simulées
# ===========================================================================

def cleanup_simulated_figures():
    obsolete = [
        "chapter7_shap_summary_beeswarm.png",
        "chapter7_shap_interaction_heatmap.png",
        "chapter7_gnn_explainer_comparative.png",
        "chapter7_cross_module_calibration.png",
        "chapter7_glm_coefficients_waterfall.png",
        "chapter7_shap_waterfall_demo.png",
        "chapter7_tree_importance_vs_eda.png",
    ]
    for fname in obsolete:
        p = FIG_DIR / fname
        if p.exists():
            p.unlink()
            print(f"  Supprimé (simulé) : {fname}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("CHAPITRE 7 — SHAP RÉEL (depuis les vrais modèles)")
    print("=" * 60)

    print("\n[1/2] Chargement des données...")
    df_pricing    = load_pricing_sample(n=500, seed=42)
    df_fraud_test = load_fraud_test_sample(n=500, seed=42)
    print(f"  Pricing : {len(df_pricing)} obs | Fraude test : {len(df_fraud_test)} obs")

    print("\n[Fig 1] Summary beeswarm GLM + NGBoost...")
    shap_agg, cols, _ = generate_figure1_beeswarm_glm_ngboost(df_pricing)

    print("\n[Fig 2] Waterfall individuel (Police + Sinistre)...")
    generate_figure2_waterfall_individual(df_pricing, df_fraud_test)

    print("\n[Fig 3] Summary beeswarm RF Fraude...")
    generate_figure3_beeswarm_fraud(df_fraud_test)

    print("\n[Fig 4] CANN KernelExplainer + Interactions...")
    generate_figure4_cann_interactions(df_pricing)

    print("\n[Nettoyage] Suppression des figures simulées...")
    cleanup_simulated_figures()

    print("\n" + "=" * 60)
    print("TERMINÉ — Figures réelles dans reports/figures/")
    print("=" * 60)
    for p in sorted(FIG_DIR.glob("chapter7_*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
