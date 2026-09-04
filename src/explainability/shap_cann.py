"""
src/explainability/shap_cann.py

Calcule les valeurs SHAP (et interactions approximées) pour le CANN PyTorch
(GroupInteractionNet) via shap.KernelExplainer.
"""

from pathlib import Path
import torch
import numpy as np
import pandas as pd
import shap
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"

from src.pricing.cann import GroupInteractionNet

CANN_CONTINUOUS_COLS = ["VehPower_norm", "VehAge_norm", "VehGas_code"]
CANN_BRAND_COL = "VehBrand_code"
CANN_ALL_COLS = CANN_CONTINUOUS_COLS + [CANN_BRAND_COL, "glm_log_pred"]

CANN_DISPLAY_NAMES = {
    "VehPower_norm": "VehPower",
    "VehAge_norm": "VehAge",
    "VehGas_code": "VehGas",
    "VehBrand_code": "VehBrand",
    "glm_log_pred": "GLM Baseline",
}

def _load_cann_model():
    """Charge le GroupInteractionNet depuis models/cann_group_interaction.pt."""
    model = GroupInteractionNet(
        n_continuous=3,
        brand_cardinality=11,
        embedding_dim=2,
        hidden_dim=20
    )
    checkpoint = torch.load(
        MODELS_DIR / "cann_group_interaction.pt",
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def _cann_predict_fn(X_array: np.ndarray, model):
    results = []
    with torch.no_grad():
        for i in range(len(X_array)):
            row = X_array[i]
            cont = torch.tensor(row[:3], dtype=torch.float32).unsqueeze(0)
            brand = torch.tensor([row[3]], dtype=torch.long)
            glm_lp = torch.tensor([row[4]], dtype=torch.float32)
            log_mu = model(cont, brand, glm_lp)
            results.append(log_mu.item())
    return np.array(results)

def compute_shap_cann(df_sample: pd.DataFrame, n_background: int = 50, n_explain: int = 100):
    model = _load_cann_model()
    
    # Check if we need to standardize VehGas_code as the backend does
    try:
        cann_stats = joblib.load(MODELS_DIR / "cann_stats.pkl")
        df_sample = df_sample.copy()
        df_sample["VehGas_code"] = (df_sample["VehGas_code"] - cann_stats["VehGas"]["mean"]) / cann_stats["VehGas"]["std"]
    except:
        pass

    X_all = df_sample[CANN_ALL_COLS].values.astype(float)

    rng = np.random.default_rng(42)
    idx_bg  = rng.choice(len(X_all), min(n_background, len(X_all)), replace=False)
    idx_exp = rng.choice(len(X_all), min(n_explain, len(X_all)), replace=False)

    X_background = X_all[idx_bg]
    X_explain    = X_all[idx_exp]

    def predict_fn(X_arr):
        return _cann_predict_fn(X_arr, model)

    explainer   = shap.KernelExplainer(predict_fn, X_background)
    shap_values = explainer.shap_values(X_explain, nsamples=64, silent=True)

    feature_names = [CANN_DISPLAY_NAMES.get(c, c) for c in CANN_ALL_COLS]
    return shap_values, feature_names, X_explain, float(explainer.expected_value)

def compute_shap_interaction_approx(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    sv = pd.DataFrame(shap_values, columns=feature_names)
    interaction_matrix = sv.corr() * np.outer(sv.std(), sv.std())
    return interaction_matrix
