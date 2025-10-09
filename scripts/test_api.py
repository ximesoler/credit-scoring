import requests
import json

# Test health check
response = requests.get("http://localhost:8000/health")
print("="*60)
print("Health Check:")
print("="*60)
print(json.dumps(response.json(), indent=2))
print()

# Test scoring - simplified
sample_data = {
    "fico_mid": 720,
    "dti": 25.5,
    "home_ownership": "MORTGAGE",
    "verification_status": "Verified",
    "inq_last_6mths_bin": "1",
    "emp_length_yrs": 5,
    "loan_to_income_bin": "low",
    "install_to_income_bin": "medium",
    "avg_cur_bal_bin": "high",
    "acc_open_past_24mths": 3,
    "bc_open_to_buy": 15000.0,
    "mo_sin_old_rev_tl_op": 120.0,
    "mo_sin_rcnt_rev_tl_op": 6.0,
    "mo_sin_rcnt_tl": 3.0,
    "mort_acc": 1,
    "mths_since_recent_bc": 2.0,
    "mths_since_recent_inq": 4.0,
    "num_actv_rev_tl": 8,
    "num_tl_op_past_12m": 2,
    "open_rv_24m": 1,
    "total_bc_limit": 25000.0
}

response = requests.post("http://localhost:8000/score", json=sample_data)
print("="*60)
print("Scoring Result:")
print("="*60)
print(json.dumps(response.json(), indent=2))
print()

# Show what the response contains
result = response.json()
print("="*60)
print("Summary:")
print("="*60)
print(f"Decision:              {result['decision']}")
print(f"Default Probability:   {result['default_probability']:.4f}")
print(f"Confidence Score:      {result['confidence_score']:.4f}")