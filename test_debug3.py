import sys
sys.path.insert(0, '.')
from backend.app.services.explainability_service import _build_design_matrix_row, _get_glm_coefficients
from backend.app.schemas.pricing import PricingRequest

request = PricingRequest(
    veh_power=6, veh_age=5, driv_age=22, bonus_malus=75,
    veh_brand='B2', veh_gas='Regular', region='Ile-de-France',
    area='A', density=5000, exposure=1.0
)

X = _build_design_matrix_row(request)
params = _get_glm_coefficients()

print("Design matrix columns:", X.columns.tolist())
print("Design matrix values:")
for col in X.columns:
    print(f"  {col}: {X[col].iloc[0]}")

print("\nModel params:")
for name, val in params.items():
    print(f"  {name}: {val:.4f}")

print("\nMatching params to X columns:")
for param_name, coef in params.items():
    if param_name == "Intercept":
        continue
    if param_name in X.columns:
        x_val = X[param_name].iloc[0]
        print(f"  MATCH: {param_name} -> x_val={x_val}, coef={coef:.4f}")
    else:
        print(f"  MISSING: {param_name}")

print("\nX columns not in params:")
for col in X.columns:
    if col not in params.index and col != "Intercept":
        print(f"  EXTRA: {col}")