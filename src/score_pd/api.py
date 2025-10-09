"""
Credit Scoring REST API

FastAPI-based REST API for real-time credit scoring.

Usage:
    uvicorn src.score_pd.api:app --reload --port 8000

Author: [Tu nombre]
Date: October 2025
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import uvicorn
from datetime import datetime

from .inference import CreditScorer

# Initialize FastAPI app
app = FastAPI(
    title="Credit Scoring API",
    description="Production-ready API for credit risk assessment",
    version="1.0.0"
)

# Initialize scorer (loads models on startup)
scorer = CreditScorer(model_dir="models")


# Request/Response Models
class ApplicantFeatures(BaseModel):
    """Schema for credit application features."""
    
    fico_mid: float = Field(..., ge=300, le=850, description="FICO credit score")
    dti: float = Field(..., ge=0, le=100, description="Debt-to-income ratio (%)")
    home_ownership: str = Field(..., description="Home ownership status")
    verification_status: str = Field(..., description="Income verification status")
    inq_last_6mths_bin: str = Field(..., description="Recent credit inquiries (binned)")
    emp_length_yrs: Optional[float] = Field(None, ge=0, description="Years of employment")
    
    loan_to_income_bin: Optional[str] = None
    install_to_income_bin: Optional[str] = None
    avg_cur_bal_bin: Optional[str] = None
    acc_open_past_24mths: Optional[int] = None
    bc_open_to_buy: Optional[float] = None
    mo_sin_old_rev_tl_op: Optional[float] = None
    mo_sin_rcnt_rev_tl_op: Optional[float] = None
    mo_sin_rcnt_tl: Optional[float] = None
    mort_acc: Optional[int] = None
    mths_since_recent_bc: Optional[float] = None
    mths_since_recent_inq: Optional[float] = None
    num_actv_rev_tl: Optional[int] = None
    num_tl_op_past_12m: Optional[int] = None
    open_rv_24m: Optional[int] = None
    total_bc_limit: Optional[float] = None
    
    @validator('home_ownership')
    def validate_home_ownership(cls, v):
        valid_values = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
        if v.upper() not in valid_values:
            raise ValueError(f"home_ownership must be one of {valid_values}")
        return v.upper()
    
    class Config:
        schema_extra = {
            "example": {
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
                "bc_open_to_buy": 15000,
                "mo_sin_old_rev_tl_op": 120,
                "mo_sin_rcnt_rev_tl_op": 6,
                "mo_sin_rcnt_tl": 3,
                "mort_acc": 1,
                "mths_since_recent_bc": 2,
                "mths_since_recent_inq": 4,
                "num_actv_rev_tl": 8,
                "num_tl_op_past_12m": 2,
                "open_rv_24m": 1,
                "total_bc_limit": 25000
            }
        }


class ScoringResponse(BaseModel):
    """Schema for scoring response."""
    
    application_id: int
    decision: str
    default_probability: float
    confidence_score: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# API Endpoints
@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "service": "Credit Scoring API",
        "status": "healthy",
        "version": "1.0.0",
        "model": "CatBoost",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": scorer.model is not None,
        "preprocessor_loaded": scorer.preprocessor is not None,
        "calibrator_loaded": scorer.calibrator is not None,
        "features_count": len(scorer.feature_list) if scorer.feature_list else 0,
        "approval_threshold": scorer.approval_threshold
    }


@app.post("/score", response_model=ScoringResponse)
def score_application(applicant: ApplicantFeatures):
    """
    Score a single credit application.
    
    Returns credit decision, probability, and risk details.
    """
    try:
        # Convert Pydantic model to dict
        applicant_data = applicant.dict()
        
        # Get scoring result
        result = scorer.score_applicant(applicant_data, return_details=True)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")


@app.post("/score/batch")
def score_batch(applicants: List[ApplicantFeatures]):
    """
    Score multiple applications in batch.
    
    Returns list of scoring results.
    """
    try:
        import pandas as pd
        
        # Convert to DataFrame
        applicants_data = pd.DataFrame([app.dict() for app in applicants])
        
        # Score batch
        results = scorer.score_applicant(applicants_data, return_details=True)
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring error: {str(e)}")


@app.post("/threshold/update")
def update_threshold(new_threshold: float):
    """
    Update approval threshold (for business policy changes).
    
    Args:
        new_threshold: New probability threshold (0-1)
    """
    try:
        old_threshold = scorer.approval_threshold
        scorer.set_approval_threshold(new_threshold)
        
        return {
            "message": "Threshold updated successfully",
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model/info")
def model_info():
    """Get model metadata and configuration."""
    return {
        "model_type": "CatBoost Classifier",
        "features_count": len(scorer.feature_list),
        "approval_threshold": scorer.approval_threshold,
        "calibration": "Isotonic Regression"
    }


if __name__ == "__main__":
    # Run with: python -m src.score_pd.api
    uvicorn.run(app, host="0.0.0.0", port=8000)