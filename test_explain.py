import sys
sys.path.insert(0, '.')
from backend.app.services.explainability_service import explain_pricing
from backend.app.schemas.pricing import PricingRequest

request = PricingRequest(
    veh_power=6,
    veh_age=5,
    driv_age=35,
    bonus_malus=50,
    veh_brand='B1',
    veh_gas='Regular',
    region='Ile-de-France',
    area='A',
    density=5000,
    exposure=1.0
)

result = explain_pricing(request)
print(result)