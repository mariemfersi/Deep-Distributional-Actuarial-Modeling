import sys
sys.path.insert(0, '.')
from backend.app.services.explainability_service import explain_pricing
from backend.app.schemas.pricing import PricingRequest

# Test with non-reference categories
request = PricingRequest(
    veh_power=6,
    veh_age=5,
    driv_age=22,  # 21-25 bucket (not reference)
    bonus_malus=75,  # 61-80 bucket (not reference)
    veh_brand='B2',  # not reference (B1 is reference)
    veh_gas='Regular',  # not reference (Diesel is reference)
    region='Ile-de-France',  # not reference (Aquitaine is reference)
    area='A',
    density=5000,
    exposure=1.0
)

result = explain_pricing(request)
print("Result with non-reference categories:")
import json
print(json.dumps(result, indent=2))

# Also test reference case
request2 = PricingRequest(
    veh_power=6,
    veh_age=5,
    driv_age=19,  # 18-20 bucket (reference)
    bonus_malus=50,  # 50-60 bucket (reference)
    veh_brand='B1',  # reference
    veh_gas='Diesel',  # reference
    region='Aquitaine',  # reference
    area='A',
    density=5000,
    exposure=1.0
)

result2 = explain_pricing(request2)
print("\nResult with all reference categories:")
print(json.dumps(result2, indent=2))