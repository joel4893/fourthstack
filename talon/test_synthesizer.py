import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from core.synthesizer import synthesize, validate_dataframe

# Generate test data — same as research
np.random.seed(42)
n = 500

def generate_hours(n):
    hours = []
    for _ in range(n):
        c = np.random.choice(['midnight','lunch','evening','other'],
                              p=[0.25,0.20,0.30,0.25])
        if c == 'midnight': h = int(np.clip(np.random.normal(1,1),0,3))  # noqa: E701
        elif c == 'lunch':  h = int(np.clip(np.random.normal(12,1),10,14))  # noqa: E701
        elif c == 'evening':h = int(np.clip(np.random.normal(19,1.5),16,23))  # noqa: E701
        else:               h = np.random.randint(0,24)  # noqa: E701
        hours.append(h)
    return hours

test_data = pd.DataFrame({
    'transaction_id':    range(1, n+1),
    'amount':            np.round(np.random.lognormal(3.5, 1.2, n), 2),
    'merchant_category': np.random.choice(
        ['groceries','transport','dining','entertainment','utilities'],
        size=n, p=[0.30,0.25,0.20,0.15,0.10]
    ),
    'transaction_hour':  generate_hours(n),
    'is_fraud':          np.random.choice([0,1], size=n, p=[0.97,0.03]),
    'customer_age':      np.random.randint(18, 75, n),
    'account_balance':   np.round(np.random.lognormal(7, 1, n), 2)
})

print("── Testing synthesizer.py ────────────────────────")
print(f"Input: {len(test_data)} rows")

# Test validation
print("\n── Validation test ───────────────────────────────")
v = validate_dataframe(test_data)
print(f"Valid: {v['valid']}")
print(f"Errors: {v['errors']}")

# Test synthesis
print("\n── Running synthesis pipeline ────────────────────")
print("(takes ~4 mins)")
result = synthesize(test_data)

if result['success']:
    print("\n── Result ────────────────────────────────────────")
    print(f"Success: {result['success']}")
    print(f"Synthetic rows: {len(result['synthetic'])}")
    print()
    for k, v in result['fidelity'].items():
        print(f"  {k:<25} {v}")
else:
    print(f"Failed: {result['errors']}")