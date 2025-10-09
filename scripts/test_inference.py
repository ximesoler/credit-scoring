"""
Test script for credit scoring inference system.

Tests both the inference module and the API locally.
"""

import sys
sys.path.append('.')

from src.score_pd.inference import CreditScorer
import pandas as pd
import requests
import json


def test_inference_module():
    """Test the inference module directly."""
    print("\n" + "="*60)
    print("TEST 1: Direct Inference Module")
    print("="*60)
    
    # Initialize scorer
    scorer = CreditScorer(model_dir="models")
    
    # Test cases
    test_cases = [
        {
            "name": "Prime Applicant",
            "data": {
                'fico_mid': 780,
                'dti': 15.0,
                'home_ownership': 'OWN',
                'verification_status': 'Verified',
                'inq_last_6mths_bin': '0',
                'emp_length_yrs': 10,
            }
        },
        {
            "name": "Subprime Applicant",
            "data": {
                'fico_mid': 620,
                'dti': 42.0,
                'home_ownership': 'RENT',
                'verification_status': 'Not Verified',
                'inq_last_6mths_bin': '4+',
                'emp_length_yrs': 1,
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        result = scorer.score_applicant(test_case['data'])
        print(f"  Decision: {result['decision']}")
        print(f"  Probability: {result['default_probability']:.4f}")
        print(f"  Risk Tier: {result['risk_tier']}")
        print(f"  Recommended APR: {result['recommended_apr_range']}")


def test_api():
    """Test the API endpoints (requires running server)."""
    print("\n" + "="*60)
    print("TEST 2: REST API")
    print("="*60)
    print("\nNOTE: Start API first with: uvicorn src.score_pd.api:app --reload")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test health check
        response = requests.get(f"{base_url}/health")
        print(f"\n✓ Health Check: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        
        # Test scoring
        sample_request = {
            "fico_mid": 720,
            "dti": 25.5,
            "home_ownership": "MORTGAGE",
            "verification_status": "Verified",
            "inq_last_6mths_bin": "1",
            "emp_length_yrs": 5
        }
        
        response = requests.post(f"{base_url}/score", json=sample_request)
        print(f"\n✓ Score Endpoint: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        
    except requests.ConnectionError:
        print("\n⚠️  API server not running. Start with:")
        print("   uvicorn src.score_pd.api:app --reload --port 8000")


if __name__ == "__main__":
    test_inference_module()
    test_api()