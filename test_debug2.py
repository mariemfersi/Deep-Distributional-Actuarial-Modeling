import joblib
model = joblib.load('models/glm_poisson.pkl')

# Extract region categories from model params
region_params = [p for p in model.params.index if p.startswith('C(Region)')]
print('Model regions:')
for r in region_params:
    cat = r.split('[T.')[1].rstrip(']')
    print(f'  {cat}')

# Check VehGas
vehgas_params = [p for p in model.params.index if 'VehGas' in p]
print('\nModel VehGas params:')
for v in vehgas_params:
    print(f'  {v}')

# Check VehBrand
vehbrand_params = [p for p in model.params.index if 'VehBrand' in p]
print('\nModel VehBrand params:')
for v in vehbrand_params:
    print(f'  {v}')

# Check FEATURE_CATEGORIES in explainability
from backend.app.services.explainability_service import FEATURE_CATEGORIES
print('\nFEATURE_CATEGORIES:')
for k, v in FEATURE_CATEGORIES.items():
    if v:
        print(f'  {k}: {v}')