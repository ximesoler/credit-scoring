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
    "fico_mid": 300,
    "dti": 25.5,
    "home_ownership": "MORTGAGE",
    "verification_status": "verified",
    "inq_last_6mths_bin": "1_0_2_0",
    "emp_length_yrs": 5,
    "acc_open_past_24mths": 10,
    "all_util": 45.2,
    "avg_cur_bal_bin": "30000_0_inf",
    "bc_open_to_buy": 15000,
    "il_util": 65.0,
    "inq_fi": 2,
    "inq_last_12m_bin": "1_0_4_0",
    "install_to_income_bin": "0_00101_0_02",
    "loan_to_income_bin": "0_3_inf",
    "max_bal_bc": 8000,
    "mo_sin_old_rev_tl_op": 120,
    "mo_sin_rcnt_rev_tl_op": 6,
    "mo_sin_rcnt_tl": 3,
    "mort_acc": 1,
    "mths_since_rcnt_il": 12,
    "mths_since_recent_bc": 2,
    "mths_since_recent_inq": 4,
    "num_actv_rev_tl": 8,
    "num_tl_op_past_12m": 2,
    "open_acc_6m": 1,
    "open_il_24m": 2,
    "open_rv_12m": 1,
    "open_rv_24m": 1,
    "total_bc_limit": 25000
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