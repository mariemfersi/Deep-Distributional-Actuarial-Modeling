"""
Génère les visualisations du Chapitre 4 — Module Tarification (Pricing / CANN).

Sorties : reports/figures/chapter4_*.png

Le script charge les modèles déjà entraînés depuis models/ lorsque disponibles :
- glm_poisson.pkl
- glm_gamma.pkl
- ngboost_severity.pkl
- cann_group_interaction.pt

Il ne réentraîne pas le CANN afin de conserver les résultats validés du mémoire.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import spearmanr, norm
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.pricing.data import build_pricing_dataset, train_valid_test_split, get_severity_subset
from src.pricing.features import build_features
from src.pricing.models import (
    fit_glm_poisson,
    fit_glm_gamma,
    fit_ngboost_severity,
    predict_frequency,
    predict_severity,
    NGBOOST_FEATURES,
)
from src.pricing.cann import GroupInteractionNet, GroupDataset
from src.pricing.evaluate import compute_gini_index, compute_lorenz_curve

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

COLORS = {
    "glm": "#2A6FBB",
    "cann": "#D95F02",
    "green": "#2F855A",
    "purple": "#6B46C1",
    "gray": "#666666",
    "light_blue": "#E8F1FB",
    "light_orange": "#FFF2E0",
}


def save_fig(name: str):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"OK {path.relative_to(PROJECT_ROOT)}")


def add_box(ax, xy, text, width=2.1, height=0.75, color="#FFFFFF", edge="#333333", fontsize=9.5, lw=1.5):
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=lw,
        edgecolor=edge,
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight="bold")
    return box


def add_arrow(ax, start, end, color="#555555", rad=0.0, lw=1.5, style="->"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def load_or_fit_models(train: pd.DataFrame, train_sev: pd.DataFrame):
    glm_path = MODELS_DIR / "glm_poisson.pkl"
    gamma_path = MODELS_DIR / "glm_gamma.pkl"
    ngboost_path = MODELS_DIR / "ngboost_severity.pkl"

    if glm_path.exists():
        glm_model = joblib.load(glm_path)
        print("GLM Poisson chargé.")
    else:
        glm_model = fit_glm_poisson(train)
        joblib.dump(glm_model, glm_path)
        print("GLM Poisson entraîné et sauvegardé.")

    if gamma_path.exists():
        gamma_model = joblib.load(gamma_path)
        print("GLM Gamma chargé.")
    else:
        gamma_model = fit_glm_gamma(train_sev)
        joblib.dump(gamma_model, gamma_path)
        print("GLM Gamma entraîné et sauvegardé.")

    if ngboost_path.exists():
        ngboost_model = joblib.load(ngboost_path)
        print("NGBoost chargé.")
    else:
        ngboost_model = fit_ngboost_severity(train_sev, n_estimators=300)
        joblib.dump(ngboost_model, ngboost_path)
        print("NGBoost entraîné et sauvegardé.")

    return glm_model, gamma_model, ngboost_model


def load_cann_model(device: str):
    cann_path = MODELS_DIR / "cann_group_interaction.pt"
    if not cann_path.exists():
        raise FileNotFoundError(
            "Modèle CANN introuvable : models/cann_group_interaction.pt. "
            "Relancer notebooks/03_pricing_cann_exploration.ipynb pour le sauvegarder."
        )

    model = GroupInteractionNet(n_continuous=3, brand_cardinality=11, embedding_dim=2, hidden_dim=20).to(device)
    model.load_state_dict(torch.load(cann_path, map_location=device))
    model.eval()
    print("CANN ciblé chargé.")
    return model


def prepare_predictions(df: pd.DataFrame, glm_model, cann_model, device: str) -> pd.DataFrame:
    """Ajoute les prédictions GLM et CANN au DataFrame fourni."""
    out = df.copy()
    out["glm_freq"] = predict_frequency(glm_model, out).to_numpy()
    out["glm_mu"] = out["glm_freq"] * out["Exposure"]
    out["glm_log_pred"] = np.log(np.clip(out["glm_freq"].to_numpy(), 1e-12, None))
    out["log_mu_glm"] = np.log(np.clip(out["glm_mu"].to_numpy(), 1e-12, None))

    group_continuous_cols = ["VehPower_norm", "VehAge_norm", "VehGas_code"]
    dataset = GroupDataset(out, group_continuous_cols, "VehBrand_code")
    loader = DataLoader(dataset, batch_size=8192, shuffle=False)

    cann_mu_parts = []
    with torch.no_grad():
        for batch in loader:
            continuous = batch["continuous"].to(device)
            brand_code = batch["brand_code"].to(device)
            log_mu_glm = batch["log_mu_glm"].to(device)
            log_mu_cann = cann_model(continuous, brand_code, log_mu_glm)
            mu_cann = torch.exp(torch.clamp(log_mu_cann, min=-20.0, max=10.0))
            cann_mu_parts.append(mu_cann.cpu().numpy())

    out["cann_mu"] = np.concatenate(cann_mu_parts)
    out["cann_freq"] = out["cann_mu"] / out["Exposure"]
    return out


def poisson_deviance_residuals(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    safe_y = np.where(y > 0, y, 1.0)
    unit_dev = 2 * (np.where(y > 0, y * np.log(safe_y / mu), 0.0) - (y - mu))
    unit_dev = np.maximum(unit_dev, 0.0)
    return np.sign(y - mu) * np.sqrt(unit_dev)


def build_lift_by_score(df: pd.DataFrame, score_col: str, n_bins: int = 10) -> pd.DataFrame:
    tmp = df[["ClaimNb", "Exposure", score_col]].copy()
    tmp["decile"] = pd.qcut(tmp[score_col], q=n_bins, labels=False, duplicates="drop")
    lift = tmp.groupby("decile", observed=True).agg(
        claims=("ClaimNb", "sum"),
        exposure=("Exposure", "sum"),
        pred=(score_col, "sum"),
        n=("ClaimNb", "size"),
    )
    lift["observed_freq"] = lift["claims"] / lift["exposure"]
    lift["predicted_freq"] = lift["pred"] / lift["exposure"]
    lift = lift.reset_index()
    lift["decile"] = lift["decile"] + 1
    return lift


def figure_cann_architecture():
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(7, 6.65, "Architecture CANN retenue : GLM Poisson + réseau résiduel ciblé",
            ha="center", va="center", fontsize=16, fontweight="bold")

    add_box(ax, (1.6, 4.9), "Variables tarifaires\ncomplètes", width=2.2, height=0.85,
            color="#F7F7F7", edge="#444")
    add_box(ax, (4.0, 5.45), "GLM Poisson\nlog-link + offset", width=2.1, height=0.85,
            color=COLORS["light_blue"], edge=COLORS["glm"])
    add_box(ax, (6.6, 5.45), "$\\log(\\mu_{GLM})$\ncomposante fixe", width=2.0, height=0.85,
            color="#FFFFFF", edge=COLORS["glm"])

    add_box(ax, (1.6, 2.2), "Sous-groupe ciblé\nVehPower, VehAge,\nVehGas, VehBrand", width=2.35, height=1.05,
            color="#F7F7F7", edge="#444", fontsize=9)
    add_box(ax, (4.0, 2.75), "Embedding\nVehBrand (dim. 2)", width=2.1, height=0.85,
            color="#F3E8FF", edge=COLORS["purple"])
    add_box(ax, (4.0, 1.55), "Variables continues\nnormalisées", width=2.1, height=0.85,
            color="#F3E8FF", edge=COLORS["purple"])
    add_box(ax, (6.4, 2.15), "MLP résiduel\n20 → 10 → 1\nactivation tanh", width=2.2, height=1.05,
            color=COLORS["light_orange"], edge=COLORS["cann"], fontsize=9)
    add_box(ax, (8.85, 2.15), "$f_{NN}(x)$\nrésidu appris", width=1.8, height=0.85,
            color="#FFFFFF", edge=COLORS["cann"])

    add_box(ax, (9.2, 4.35), "Addition\n(skip connection)", width=2.05, height=0.85,
            color="#E6F4EA", edge=COLORS["green"])
    ax.text(9.2, 3.62, "$\\log(\\mu_{CANN}) = \\log(\\mu_{GLM}) + f_{NN}(x)$",
            ha="center", va="center", fontsize=12, color="#222")
    add_box(ax, (11.9, 4.35), "Exponentielle", width=1.75, height=0.75,
            color="#FFFFFF", edge=COLORS["green"])
    add_box(ax, (12.0, 2.95), "Fréquence CANN\n$\\lambda_{CANN}$", width=2.0, height=0.85,
            color="#E6F4EA", edge=COLORS["green"])
    add_box(ax, (12.0, 1.65), "Prime pure\n$\\lambda_{CANN} \\times E[S|x]$", width=2.2, height=0.85,
            color="#FFF2CC", edge="#B7791F", fontsize=9)

    add_arrow(ax, (2.7, 4.9), (3.0, 5.32), color="#444")
    add_arrow(ax, (5.05, 5.45), (5.6, 5.45), color=COLORS["glm"])
    add_arrow(ax, (7.6, 5.35), (8.3, 4.65), color=COLORS["glm"], rad=-0.12)

    add_arrow(ax, (2.8, 2.35), (3.0, 2.7), color="#444")
    add_arrow(ax, (2.8, 2.05), (3.0, 1.6), color="#444")
    add_arrow(ax, (5.05, 2.75), (5.35, 2.32), color=COLORS["purple"])
    add_arrow(ax, (5.05, 1.55), (5.35, 1.98), color=COLORS["purple"])
    add_arrow(ax, (7.5, 2.15), (7.95, 2.15), color=COLORS["cann"])
    add_arrow(ax, (9.25, 2.45), (8.95, 3.9), color=COLORS["cann"], rad=0.18)
    add_arrow(ax, (10.25, 4.35), (11.0, 4.35), color=COLORS["green"])
    add_arrow(ax, (11.9, 3.95), (11.95, 3.4), color=COLORS["green"])
    add_arrow(ax, (11.95, 2.53), (11.95, 2.08), color="#B7791F")

    ax.text(5.8, 0.62,
            "Initialisation : dernière couche du réseau à zéro → au démarrage, CANN = GLM.\n"
            "Le réseau n'apprend qu'une correction locale sur l'interaction véhicule/marque, le GLM restant la référence interprétable.",
            ha="center", va="center", fontsize=9.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7F7F7", edgecolor="#BBBBBB"))

    save_fig("chapter4_01_architecture_cann.png")


def figure_lift_curves(test_pred: pd.DataFrame):
    test_pred = test_pred.copy()
    # Déciles construits sur la fréquence annuelle prédite (lambda) — pouvoir discriminant pur
    lift_glm = build_lift_by_score(test_pred, "glm_freq")
    lift_cann = build_lift_by_score(test_pred, "cann_freq")

    portfolio_freq = test_pred["ClaimNb"].sum() / test_pred["Exposure"].sum()
    lift_glm["observed_lift"] = lift_glm["observed_freq"] / portfolio_freq
    lift_cann["observed_lift"] = lift_cann["observed_freq"] / portfolio_freq
    lift_glm["predicted_lift"] = lift_glm["predicted_freq"] / portfolio_freq
    lift_cann["predicted_lift"] = lift_cann["predicted_freq"] / portfolio_freq

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1.0, color="#999999", linestyle="--", linewidth=1.2, label="Moyenne portefeuille")
    ax.plot(lift_glm["decile"], lift_glm["observed_lift"], marker="o", color=COLORS["glm"],
            linewidth=2, label="GLM — lift observé")
    ax.plot(lift_cann["decile"], lift_cann["observed_lift"], marker="o", color=COLORS["cann"],
            linewidth=2, label="CANN ciblé — lift observé")
    ax.plot(lift_glm["decile"], lift_glm["predicted_lift"], linestyle=":", color=COLORS["glm"],
            linewidth=1.8, label="GLM — lift prédit")
    ax.plot(lift_cann["decile"], lift_cann["predicted_lift"], linestyle=":", color=COLORS["cann"],
            linewidth=1.8, label="CANN — lift prédit")

    ax.set_title("Courbes de lift fréquence — GLM vs CANN ciblé", fontsize=14, fontweight="bold")
    ax.set_xlabel("Décile de risque prédit (1 = plus faible, 10 = plus élevé)")
    ax.set_ylabel("Lift de fréquence observée / moyenne portefeuille")
    ax.set_xticks(range(1, 11))
    ax.legend(loc="upper left", frameon=True)
    save_fig("chapter4_02_lift_glm_vs_cann.png")


def figure_lorenz_gini(test_pred: pd.DataFrame):
    y = test_pred["ClaimNb"].to_numpy()
    exposure = test_pred["Exposure"].to_numpy()
    # Gini en tarification : tri par fréquence prédite (lambda), pondéré par l'exposition
    glm_freq = test_pred["glm_freq"].to_numpy()
    cann_freq = test_pred["cann_freq"].to_numpy()

    lorenz_glm = compute_lorenz_curve(y, glm_freq, exposure)
    lorenz_cann = compute_lorenz_curve(y, cann_freq, exposure)
    gini_glm = compute_gini_index(y, glm_freq, exposure)
    gini_cann = compute_gini_index(y, cann_freq, exposure)

    fig, ax = plt.subplots(figsize=(8.2, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Diagonale")
    ax.plot(lorenz_glm["cum_exposure"], lorenz_glm["cum_claims"], color=COLORS["glm"], linewidth=2.2,
            label=f"GLM — Gini = {gini_glm:.3f}")
    ax.plot(lorenz_cann["cum_exposure"], lorenz_cann["cum_claims"], color=COLORS["cann"], linewidth=2.2,
            label=f"CANN ciblé — Gini = {gini_cann:.3f}")

    ax.set_title("Courbes de Lorenz ordonnées par risque prédit", fontsize=14, fontweight="bold")
    ax.set_xlabel("Part cumulée de l'exposition")
    ax.set_ylabel("Part cumulée des sinistres observés")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True)
    save_fig("chapter4_03_lorenz_gini_glm_vs_cann.png")


def figure_premium_profile_intervals(test_pred: pd.DataFrame, gamma_model, ngboost_model):
    test_pred = test_pred.copy()
    test_pred["severity_glm"] = predict_severity(gamma_model, test_pred)
    test_pred["pure_premium_cann"] = test_pred["cann_freq"] * test_pred["severity_glm"]

    quantiles = [0.10, 0.50, 0.90, 0.99]
    labels = ["Profil faible\nP10", "Profil médian\nP50", "Profil élevé\nP90", "Profil extrême\nP99"]
    sorted_idx = np.argsort(test_pred["pure_premium_cann"].to_numpy())
    chosen_idx = [sorted_idx[int(q * (len(sorted_idx) - 1))] for q in quantiles]
    profiles = test_pred.iloc[chosen_idx].copy()

    X = profiles[NGBOOST_FEATURES].values
    dist = ngboost_model.pred_dist(X)
    severity_q = {
        "p05": dist.dist.ppf(0.05),
        "p25": dist.dist.ppf(0.25),
        "p50": dist.dist.ppf(0.50),
        "p75": dist.dist.ppf(0.75),
        "p95": dist.dist.ppf(0.95),
    }
    freq = profiles["cann_freq"].to_numpy()
    premium_q = {k: np.asarray(v) * freq for k, v in severity_q.items()}
    premium_mean = np.asarray(dist.mean()) * freq

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 6))

    # Bande 90% puis intervalle interquartile : fan chart discret par profil type.
    ax.vlines(x, premium_q["p05"], premium_q["p95"], color=COLORS["cann"], linewidth=9, alpha=0.22,
              label="Intervalle prédictif 90%")
    ax.vlines(x, premium_q["p25"], premium_q["p75"], color=COLORS["cann"], linewidth=15, alpha=0.42,
              label="Intervalle interquartile")
    ax.scatter(x, premium_q["p50"], color=COLORS["cann"], s=70, zorder=3, label="Médiane")
    ax.scatter(x, premium_mean, color="#111111", s=45, zorder=4, marker="D", label="Moyenne")

    for i, row in enumerate(profiles.itertuples()):
        detail = f"âge {int(row.DrivAge)} | BM {int(row.BonusMalus)}\n{row.VehBrand}, {row.VehGas}, veh {int(row.VehAge)} ans"
        ax.text(i, premium_q["p95"][i] * 1.08, detail, ha="center", va="bottom", fontsize=8, color="#444")

    ax.set_title("Distribution prédictive de la prime pure CANN pour profils types", fontsize=14, fontweight="bold")
    ax.set_ylabel("Prime pure annuelle (€)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=True)
    ax.text(0.5, -0.18,
            "Intervalle obtenu en multipliant la fréquence CANN par les percentiles NGBoost de sévérité.\n"
            "La fréquence est fixée au profil ; l'incertitude représentée porte sur la sévérité attritionnelle.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#555")
    save_fig("chapter4_04_profils_prime_predite_intervalles.png")


def figure_deviance_residuals(test_pred: pd.DataFrame):
    y = test_pred["ClaimNb"].to_numpy()
    resid_glm = poisson_deviance_residuals(y, test_pred["glm_mu"].to_numpy())
    resid_cann = poisson_deviance_residuals(y, test_pred["cann_mu"].to_numpy())

    clip_min, clip_max = np.percentile(np.concatenate([resid_glm, resid_cann]), [0.5, 99.5])
    bins = np.linspace(clip_min, clip_max, 70)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})

    axes[0].hist(np.clip(resid_glm, clip_min, clip_max), bins=bins, density=True,
                 color=COLORS["glm"], alpha=0.45, label="GLM")
    axes[0].hist(np.clip(resid_cann, clip_min, clip_max), bins=bins, density=True,
                 color=COLORS["cann"], alpha=0.45, label="CANN ciblé")
    axes[0].axvline(0, color="#333333", linewidth=1.0)
    axes[0].set_title("Distribution des résidus de déviance", fontweight="bold")
    axes[0].set_xlabel("Résidu de déviance de Poisson")
    axes[0].set_ylabel("Densité")
    axes[0].legend(frameon=True)

    data = [resid_glm, resid_cann]
    parts = axes[1].violinplot(data, positions=[1, 2], showmeans=True, showextrema=False, widths=0.7)
    for body, color in zip(parts["bodies"], [COLORS["glm"], COLORS["cann"]]):
        body.set_facecolor(color)
        body.set_alpha(0.45)
    parts["cmeans"].set_color("#111111")
    axes[1].boxplot(data, positions=[1, 2], widths=0.25, showfliers=False,
                    medianprops=dict(color="#111111"), boxprops=dict(color="#444444"),
                    whiskerprops=dict(color="#444444"), capprops=dict(color="#444444"))
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["GLM", "CANN"])
    axes[1].set_title("Comparaison robuste\n(sans valeurs extrêmes)", fontweight="bold")
    axes[1].set_ylabel("Résidu")

    fig.suptitle("Résidus de déviance — GLM vs CANN ciblé", fontsize=14, fontweight="bold")
    save_fig("chapter4_05_residus_deviance_glm_vs_cann.png")


def empirical_quantile(values: np.ndarray, u: np.ndarray) -> np.ndarray:
    probs = np.linspace(0, 1, len(values))
    sorted_values = np.sort(values)
    return np.interp(u, probs, sorted_values)


def figure_frequency_severity_dependence(test_pred: pd.DataFrame):
    sev = get_severity_subset(test_pred).copy()
    x = sev["glm_freq"].to_numpy()
    y = sev["ClaimAmount_capped"].to_numpy()
    rho_s, pval = spearmanr(x, y)
    rho_gauss = 2 * np.sin(np.pi * rho_s / 6)  # inversion rho_s = 6/pi asin(rho/2)

    rng = np.random.default_rng(123)
    n = len(sev)
    z = rng.multivariate_normal(mean=[0, 0], cov=[[1, rho_gauss], [rho_gauss, 1]], size=n)
    u = norm.cdf(z)
    sim_x = empirical_quantile(x, u[:, 0])
    sim_y = empirical_quantile(y, u[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)

    hb0 = axes[0].hexbin(x, y, gridsize=38, bins="log", mincnt=1, cmap="Blues")
    axes[0].set_title("Données test : fréquence prédite vs sévérité observée", fontweight="bold")
    axes[0].set_xlabel("Fréquence annuelle prédite par le GLM")
    axes[0].set_ylabel("Sévérité observée attritionnelle (€)")
    axes[0].set_yscale("log")
    axes[0].text(0.03, 0.94, f"Spearman $\\rho$ = {rho_s:.3f}\np < 0.001",
                 transform=axes[0].transAxes, va="top", ha="left", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BBBBBB"))

    hb1 = axes[1].hexbin(sim_x, sim_y, gridsize=38, bins="log", mincnt=1, cmap="Oranges")
    axes[1].set_title("Copule gaussienne ajustée\n(signal très faible)", fontweight="bold")
    axes[1].set_xlabel("Fréquence simulée avec marginale empirique")
    axes[1].text(0.03, 0.94, f"Paramètre copule $\\rho_G$ ≈ {rho_gauss:.3f}\nCopule non retenue dans le modèle final",
                 transform=axes[1].transAxes, va="top", ha="left", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BBBBBB"))

    cbar0 = fig.colorbar(hb0, ax=axes[0], fraction=0.046, pad=0.03)
    cbar0.set_label("log10(nombre de points)")
    cbar1 = fig.colorbar(hb1, ax=axes[1], fraction=0.046, pad=0.03)
    cbar1.set_label("log10(nombre de points)")

    fig.suptitle("Diagnostic de dépendance fréquence-sévérité et copule", fontsize=14, fontweight="bold")
    save_fig("chapter4_06_dependance_freq_sev_copule.png")


def figure_embeddings_vs_actuarial_clustering(df_full: pd.DataFrame, cann_model):
    embedding = cann_model.brand_embedding.weight.detach().cpu().numpy()

    cat = df_full["VehBrand"].astype("category")
    brand_labels = list(cat.cat.categories)

    brand_stats = df_full.groupby("VehBrand", observed=True).agg(
        claims=("ClaimNb", "sum"),
        exposure=("Exposure", "sum"),
        amount=("ClaimAmount_capped", "sum"),
    ).reindex(brand_labels)
    brand_stats["observed_freq"] = brand_stats["claims"] / brand_stats["exposure"]
    brand_stats["observed_severity"] = np.where(
        brand_stats["claims"] > 0,
        brand_stats["amount"] / brand_stats["claims"],
        np.nan,
    )
    brand_stats["observed_pure_premium"] = brand_stats["amount"] / brand_stats["exposure"]

    try:
        brand_stats["cluster"] = pd.qcut(
            brand_stats["observed_pure_premium"],
            q=3,
            labels=["Risque faible", "Risque moyen", "Risque élevé"],
            duplicates="drop",
        )
    except ValueError:
        brand_stats["cluster"] = "Non classé"

    cluster_colors = {
        "Risque faible": "#2A6FBB",
        "Risque moyen": "#F0A202",
        "Risque élevé": "#D95F02",
        "Non classé": "#777777",
    }
    colors = [cluster_colors.get(str(c), "#777777") for c in brand_stats["cluster"]]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))

    axes[0].scatter(embedding[:, 0], embedding[:, 1], s=160, c=colors, edgecolor="white", linewidth=1.5)
    for i, brand in enumerate(brand_labels):
        axes[0].text(embedding[i, 0], embedding[i, 1] + 0.03, brand, ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_title("Embedding VehBrand appris par le CANN\n(projection native 2D)", fontweight="bold")
    axes[0].set_xlabel("Dimension embedding 1")
    axes[0].set_ylabel("Dimension embedding 2")

    exposure_scaled = 350 * brand_stats["exposure"] / brand_stats["exposure"].max() + 40
    axes[1].scatter(
        brand_stats["observed_freq"],
        brand_stats["observed_severity"],
        s=exposure_scaled,
        c=colors,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
    )
    for brand, row in brand_stats.iterrows():
        axes[1].text(row["observed_freq"], row["observed_severity"] * 1.02, brand,
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_title("Clustering actuariel classique par marque\n(fréquence, sévérité, exposition)", fontweight="bold")
    axes[1].set_xlabel("Fréquence observée")
    axes[1].set_ylabel("Sévérité moyenne observée (€)")

    handles = [plt.Line2D([0], [0], marker="o", color="w", label=label,
                          markerfacecolor=color, markersize=9)
               for label, color in cluster_colors.items() if label in set(map(str, brand_stats["cluster"]))]
    axes[1].legend(handles=handles, title="Classe actuarielle", frameon=True, loc="best")

    fig.suptitle("Embeddings catégoriels et segmentation actuarielle — variable VehBrand", fontsize=14, fontweight="bold")
    save_fig("chapter4_07_embeddings_vehbrand_clustering.png")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilisé : {device}")

    print("Chargement des données pricing...")
    df = build_pricing_dataset()
    df = build_features(df)
    train, valid, test = train_valid_test_split(df)
    train_sev = get_severity_subset(train)

    print(f"Train: {len(train):,} | Valid: {len(valid):,} | Test: {len(test):,}")

    print("Chargement des modèles...")
    glm_model, gamma_model, ngboost_model = load_or_fit_models(train, train_sev)
    cann_model = load_cann_model(device)

    print("Calcul des prédictions test...")
    test_pred = prepare_predictions(test, glm_model, cann_model, device)

    print("Génération des figures Chapitre 4...")
    figure_cann_architecture()
    figure_lift_curves(test_pred)
    figure_lorenz_gini(test_pred)
    figure_premium_profile_intervals(test_pred, gamma_model, ngboost_model)
    figure_deviance_residuals(test_pred)
    figure_frequency_severity_dependence(test_pred)
    figure_embeddings_vs_actuarial_clustering(df, cann_model)
    print("Terminé.")


if __name__ == "__main__":
    main()
