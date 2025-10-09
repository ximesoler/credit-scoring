"""
Credit Scoring Inference Module

This module provides a production-ready interface for loading
and executing the trained CatBoost credit risk model.

Author: [Tu nombre]
Date: October 2025
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditScorer:
    """
    Production-ready credit scoring inference class.
    
    Handles model loading, preprocessing, prediction, and business logic.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the credit scorer with trained artifacts.
        
        Args:
            model_dir: Directory containing model artifacts
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.preprocessor = None
        self.calibrator = None
        self.feature_list = None
        self.approval_threshold = 0.2
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all required model artifacts."""
        try:
            # Load CatBoost model
            from catboost import CatBoostClassifier
            model_path = self.model_dir / "catboost_model.cbm"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            self.model = CatBoostClassifier()
            self.model.load_model(str(model_path))
            logger.info(f"✓ Loaded CatBoost model from {model_path}")
            
            # Load preprocessor
            preprocessor_path = self.model_dir / "preprocessor.pkl"
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info(f"✓ Loaded preprocessor from {preprocessor_path}")
            
            # Load calibrator
            calibrator_path = self.model_dir / "calibrator_cat.pkl"
            self.calibrator = joblib.load(calibrator_path)
            logger.info(f"✓ Loaded calibrator from {calibrator_path}")
            
            # Load feature list
            feature_path = self.model_dir / "feature_list.txt"
            with open(feature_path, 'r') as f:
                self.feature_list = [line.strip() for line in f]
            logger.info(f"✓ Loaded {len(self.feature_list)} features")
            
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise
    
    def _validate_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare input data.
        
        Args:
            data: DataFrame with applicant features
            
        Returns:
            Validated DataFrame with required features
        """
        # Check for required features
        missing_features = set(self.feature_list) - set(data.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Will be imputed.")
        
        # Ensure correct column order
        data_ordered = data.reindex(columns=self.feature_list, fill_value=np.nan)
        
        return data_ordered
    
    def predict_proba(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """
        Predict default probability for applicant(s).
        
        Args:
            data: DataFrame or dict with applicant features
            
        Returns:
            Array of calibrated default probabilities
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Validate input
        data = self._validate_input(data)
        
        # Preprocess (handles categorical encoding, imputation)
        # Note: For CatBoost, we don't use preprocessor, pass raw data
        # But we need to ensure categorical features are properly typed
        cat_features = [
            "home_ownership", "inq_last_6mths_bin", 
            "install_to_income_bin", "loan_to_income_bin",
            "verification_status", "avg_cur_bal_bin"
        ]
        
        for col in cat_features:
            if col in data.columns:
                data[col] = data[col].astype(str)
        
        # Get raw predictions from CatBoost
        raw_proba = self.model.predict_proba(data)[:, 1]
        
        # Apply calibration
        calibrated_proba = self.calibrator.predict(raw_proba)
        
        return calibrated_proba
    
    def score_applicant(
        self, 
        data: Union[pd.DataFrame, Dict],
        return_details: bool = False  # Cambiamos el default a False
    ) -> Union[Dict, List[Dict]]:
        """
        Score applicant(s) and return decision with details.
        
        Args:
            data: Applicant features (dict or DataFrame)
            return_details: If True, include additional details (default: False)
            
        Returns:
            Scoring result(s) with decision and probability
        """
        # Get predictions
        probabilities = self.predict_proba(data)
        
        # Convert to DataFrame if dict input
        if isinstance(data, dict):
            data = pd.DataFrame([data])
            single_input = True
        else:
            single_input = False
        
        results = []
        for idx, prob in enumerate(probabilities):
            result = {
                "application_id": data.index[idx] if hasattr(data, 'index') else idx,
                "default_probability": float(prob),
                "decision": "APPROVED" if prob < self.approval_threshold else "DECLINED",
                "confidence_score": float(1 - abs(prob - self.approval_threshold) / self.approval_threshold)
            }
            
            results.append(result)
        
        return results[0] if single_input else results
    
    def batch_score(
        self, 
        data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Score multiple applicants in batch.
        
        Args:
            data: DataFrame with multiple applicants
            output_path: Optional path to save results
            
        Returns:
            DataFrame with scores and decisions
        """
        logger.info(f"Batch scoring {len(data)} applications...")
        
        results = self.score_applicant(data, return_details=True)
        results_df = pd.DataFrame(results)
        
        if output_path:
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        return results_df
    
    def set_approval_threshold(self, threshold: float):
        """
        Update approval threshold for business policy changes.
        
        Args:
            threshold: New probability threshold (0-1)
        """
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        logger.info(f"Updating approval threshold: {self.approval_threshold:.4f} → {threshold:.4f}")
        self.approval_threshold = threshold


# Convenience function for quick scoring
def score_application(applicant_data: Dict, model_dir: str = "models") -> Dict:
    """
    Quick scoring function for single applications.
    
    Args:
        applicant_data: Dictionary with applicant features
        model_dir: Path to model artifacts
        
    Returns:
        Scoring result with decision
    
    Example:
        >>> result = score_application({
        ...     'fico_mid': 720,
        ...     'dti': 25.5,
        ...     'home_ownership': 'MORTGAGE',
        ...     ...
        ... })
        >>> print(result['decision'])
        'APPROVED'
    """
    scorer = CreditScorer(model_dir=model_dir)
    return scorer.score_applicant(applicant_data)


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Credit Scoring Inference Module - Test")
    print("="*60)
    
    # Initialize scorer
    scorer = CreditScorer(model_dir="../models")
    
    # Test with sample applicant
    sample_applicant = {
        'fico_mid': 720,
        'dti': 25.5,
        'home_ownership': 'MORTGAGE',
        'verification_status': 'Verified',
        'inq_last_6mths_bin': '1',
        'emp_length_yrs': 5,
        'loan_to_income_bin': 'low',
        'install_to_income_bin': 'medium',
        'avg_cur_bal_bin': 'high',
        'acc_open_past_24mths': 3,
        'bc_open_to_buy': 15000,
        'mo_sin_old_rev_tl_op': 120,
        'mo_sin_rcnt_rev_tl_op': 6,
        'mo_sin_rcnt_tl': 3,
        'mort_acc': 1,
        'mths_since_recent_bc': 2,
        'mths_since_recent_inq': 4,
        'num_actv_rev_tl': 8,
        'num_tl_op_past_12m': 2,
        'open_rv_24m': 1,
        'total_bc_limit': 25000,
    }
    
    result = scorer.score_applicant(sample_applicant)
    
    print(f"\n📊 Scoring Result:")
    print(f"  Decision: {result['decision']}")
    print(f"  Default Probability: {result['default_probability']:.4f}")
    print(f"  Risk Tier: {result['risk_tier']}")
    print(f"  Recommended APR: {result['recommended_apr_range']}")
    print(f"  Key Risk Factors: {', '.join(result['key_risk_factors'])}")
    print("\n✓ Inference module working correctly!")