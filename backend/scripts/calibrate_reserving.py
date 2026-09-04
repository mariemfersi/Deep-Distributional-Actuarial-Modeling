"""
Script de calibration conforme du provisionnement Mack.

Exécute la procédure split-conformal authentique sur le portefeuille de
provisionnement et persiste les métriques mesurées (q_hat, couvertures,
largeurs) dans models/reserving_calibration.json.

Usage (dans le conteneur backend) :
    python -m scripts.calibrate_reserving

Ce script ne code jamais aucune valeur : tout est calculé à partir des
données et des sorties réelles de src.reserving.models.
"""

import json
import sys
from pathlib import Path

# Ajoute la racine du projet au path pour importer src/
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from src.reserving.calibration import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    run_conformal_calibration,
    save_calibration,
)


def main() -> None:
    print("Calibration conforme authentique du provisionnement Mack…")
    result = run_conformal_calibration(alpha=0.10, calib_frac=0.5, seed=123)

    path = save_calibration(result, DEFAULT_CALIBRATION_PATH)
    print(f"Calibration terminée → {path}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
