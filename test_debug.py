import joblib
model = joblib.load('models/glm_poisson.pkl')
print('Number of params:', len(model.params))
print('Params:')
for name, val in model.params.items():
    print(f'  {name}: {val:.4f}')

import sys
sys.path.insert(0, '.')
from backend.app.services.explainability_service import _build_design_matrix_row
from backend.app.schemas.pricing import PricingRequest

request = PricingRequest(
    veh_power=6, veh_age=5, driv_age=35, bonus_malus=50,
    veh_brand='B1', veh_gas='Regular', region='Ile-de-France',
    area='A', density=5000, exposure=1.0
)
X = _build_design_matrix_row(request)
print()
print('Design matrix columns:', len(X.columns))
print(X.columns.tolist())