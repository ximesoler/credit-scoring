"""
Credit Scoring API Constants

Feature definitions and model configuration constants.

Author: Ximena Ramos
Date: October 2025
"""

from typing import List, Dict

# Model Configuration
MODEL_TYPE = "CatBoost Classifier"
MODEL_VERSION = "1.0.0"
DEFAULT_APPROVAL_THRESHOLD = 0.2
CALIBRATION_METHOD = "Isotonic Regression"

# Feature Set - Complete list of features required by the model
FINAL_FEATURE_SET: List[str] = [
    'acc_open_past_24mths',
    'all_util',
    'avg_cur_bal_bin',
    'bc_open_to_buy',
    'dti',
    'emp_length_yrs',
    'fico_mid',
    'home_ownership',
    'il_util',
    'inq_fi',
    'inq_last_12m_bin',
    'inq_last_6mths_bin',
    'install_to_income_bin',
    'loan_to_income_bin',
    'max_bal_bc',
    'mo_sin_old_rev_tl_op',
    'mo_sin_rcnt_rev_tl_op',
    'mo_sin_rcnt_tl',
    'mort_acc',
    'mths_since_rcnt_il',
    'mths_since_recent_bc',
    'mths_since_recent_inq',
    'num_actv_rev_tl',
    'num_tl_op_past_12m',
    'open_acc_6m',
    'open_il_24m',
    'open_rv_12m',
    'open_rv_24m',
    'total_bc_limit',
    'verification_status'
]

# Required features (minimum set for API validation)
REQUIRED_FEATURES: List[str] = [
    'fico_mid',
    'dti',
    'home_ownership',
    'verification_status',
    'inq_last_6mths_bin',
    'emp_length_yrs'
]

# Categorical features
CATEGORICAL_FEATURES: List[str] = [
    'home_ownership',
    'verification_status',
    'inq_last_6mths_bin',
    'inq_last_12m_bin',
    'install_to_income_bin',
    'loan_to_income_bin',
    'avg_cur_bal_bin'
]

# Numerical features
NUMERICAL_FEATURES: List[str] = [
    f for f in FINAL_FEATURE_SET 
    if f not in CATEGORICAL_FEATURES
]

# Feature validation rules
FEATURE_CONSTRAINTS: Dict[str, Dict] = {
    'fico_mid': {'min': 300, 'max': 850, 'description': 'FICO credit score'},
    'dti': {'min': 0, 'max': 100, 'description': 'Debt-to-income ratio (%)'},
    'emp_length_yrs': {'min': 0, 'max': 50, 'description': 'Years of employment'},
    'all_util': {'min': 0, 'max': 100, 'description': 'All utilization (%)'},
    'il_util': {'min': 0, 'max': 100, 'description': 'Installment loan utilization (%)'},
}

# Valid categorical values
VALID_CATEGORICAL_VALUES: Dict[str, List[str]] = {
    'home_ownership': ['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
    'verification_status': ['Verified', 'Not Verified', 'Source Verified'],
    'inq_last_6mths_bin': ['0', '1', '2', '3', '4+'],
    'inq_last_12m_bin': ['0', '1', '2', '3', '4+'],
    'install_to_income_bin': ['low', 'medium', 'high'],
    'loan_to_income_bin': ['low', 'medium', 'high'],
    'avg_cur_bal_bin': ['low', 'medium', 'high']
}

# Model paths
MODEL_DIR = "models"
MODEL_FILENAME = "catboost_model.cbm"
PREPROCESSOR_FILENAME = "preprocessor.pkl"
CALIBRATOR_FILENAME = "calibrator_cat.pkl"
FEATURE_LIST_FILENAME = "feature_list.txt"