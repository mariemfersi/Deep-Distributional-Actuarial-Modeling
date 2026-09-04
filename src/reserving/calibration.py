"""
Calibration conforme authentique du provisionnement Mack.

Ce module exécute la procédure complète de split-conformal prediction telle
que décrite dans src/reserving/models.py (split_conformal_calibration) :

    TRAIN (Mack sur triangles observés)
    → CALIBRATION (scores de non-conformité normalisés par l'erreur standard Mack)
    → q_hat (quantile empirique de Vovk, correction taille finie)
    → TEST PREDICTION (intervalles sur le jeu de test)
    → EMPIRICAL COVERAGE (couverture mesurée sur le jeu de test)

Le résultat est persistant dans un artefact JSON (models/reserving_calibration.json)
afin que l'API puisse charger q_hat et la couverture sans refaire 143 ajustements
Mack à chaque requête. L'artefact est régénéré par ce module (offline) et par le
script backend/scripts/calibrate_reserving.py.

Les valeurs de couverture et q_hat ici sont mesurées, jamais codées en dur.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.reserving.data import load_raw_reserving_data, split_observed_future
from src.reserving.models import evaluate_mack_coverage, split_conformal_calibration

# Chemin de l'artefact de calibration (réécrit à chaque run de calibration)
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "models" / "reserving_calibration.json"


def run_conformal_calibration(
    alpha: float = 0.10,
    calib_frac: float = 0.5,
    seed: int = 123,
) -> dict[str, Any]:
    """
    Exécute la calibration conforme authentique et retourne un dictionnaire
    sérialisable de métriques mesurées (jamais codées en dur).

    Retourne un dict avec :
      - q_hat            : quantile conforme mesuré sur la calibration
      - mack_coverage    : couverture empirique Mack mesurée
      - conformal_coverage: couverture empirique conforme mesurée
      - interval widths, ecart de couverture, nombre d'observations, etc.
    """
    df = load_raw_reserving_data()
    observed, future = split_observed_future(df)

    grcodes = sorted(observed["GRCODE"].unique())

    # 1) Évaluer Mack sur chaque compagnie (triangle observé vs. réalisation future)
    mack_results: list[pd.DataFrame] = []
    n_evaluated = 0
    for grcode in grcodes:
        res = evaluate_mack_coverage(observed, future, grcode)
        if res is not None and len(res) > 0:
            mack_results.append(res)
            n_evaluated += len(res)

    if not mack_results:
        raise RuntimeError(
            "Aucun ajustement Mack valide sur le portefeuille : la calibration "
            "conforme ne peut pas être calculée."
        )

    full = pd.concat(mack_results, ignore_index=True)

    # 2) Couverture Mack brute (mesurée)
    mack_coverage = float(full["covered_90"].mean())

    # 3) Split conformal → q_hat (mesuré) + couverture conforme (mesurée)
    conformal_test, q_hat = split_conformal_calibration(
        full, alpha=alpha, calib_frac=calib_frac, seed=seed
    )
    conformal_coverage = float(conformal_test["covered_conformal"].mean())

    # 4) Largeurs d'intervalle (mesurées)
    mack_width = float((full["upper_90"] - full["lower_90"]).mean())
    conformal_width = float(
        (conformal_test["upper_conformal"] - conformal_test["lower_conformal"]).mean()
    )

    nominal = 1.0 - alpha

    result = {
        "alpha": float(alpha),
        "nominal_coverage": nominal,
        "empirical_coverage_mack": round(mack_coverage, 4),
        "empirical_coverage_conformal": round(conformal_coverage, 4),
        "coverage_error_mack": round(mack_coverage - nominal, 4),
        "coverage_error_conformal": round(conformal_coverage - nominal, 4),
        "q_hat": round(float(q_hat), 4),
        "interval_width_mack": round(mack_width, 2),
        "interval_width_conformal": round(conformal_width, 2),
        "n_companies_evaluated": len(mack_results),
        "n_companies_total": len(grcodes),
        "n_observations_mack": int(len(full)),
        "n_observations_conformal_test": int(len(conformal_test)),
        "calib_frac": float(calib_frac),
        "seed": int(seed),
        "method": "split_conformal (Vovk et al. 2005), scores normalisés par l'écart-type Mack",
        "generated_at": _now_iso(),
    }
    return result


def save_calibration(result: dict[str, Any], path: Path | None = None) -> Path:
    """Sérialise le résultat de calibration dans un artefact JSON."""
    out = path or DEFAULT_CALIBRATION_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return out


def load_calibration(path: Path | None = None) -> dict[str, Any]:
    """Charge l'artefact de calibration s'il existe, sinon None."""
    p = path or DEFAULT_CALIBRATION_PATH
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
