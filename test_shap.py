#!/usr/bin/env python
"""Test SHAP explainability for GLM pricing model."""
import sys
sys.path.insert(0, '.')

# First inspect the GLM model parameters
import joblib
model = joblib.load('models/glm_poisson.pkl')
print('Params type:', type(model.params))
print('Param names:')
for name in model.params.index:
    print(f'  {name}: {model.params[name]:.6f}')
print()

# Now test the explainability function
from backend.app.schemas.pricing import PricingRequest
from backend.app.services.explainability_service import explain_pricing

req = PricingRequest(
    veh_power=6, veh_age=5, driv_age=35, bonus_malus=60,
    veh_brand='B1', veh_gas='Diesel', region='Ile-de-France',
    area='A', density=1000, exposure=1.0
)

try:
    result = explain_pricing(req)
    print('Success!')
    print('base_value:', result['base_value'])
    for sv in result['shap_values']:
        print(f'  {sv["feature"]}: {sv["value"]:.4f}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()