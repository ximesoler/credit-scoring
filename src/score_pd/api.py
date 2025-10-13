"""
Credit Scoring REST API

FastAPI-based REST API for real-time credit scoring.

Usage:
    uvicorn src.score_pd.api:app --reload --port 8000

Author: Ximena Ramos
Date: October 2025
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import uvicorn
from datetime import datetime

from .inference import CreditScorer
from .constants import (
    MODEL_TYPE,
    MODEL_VERSION,
    FINAL_FEATURE_SET,
    REQUIRED_FEATURES,
    FEATURE_CONSTRAINTS,
    VALID_CATEGORICAL_VALUES,
    CALIBRATION_METHOD
)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Scoring API",
    description="Production-ready API for credit risk assessment",
    version=MODEL_VERSION
)

scorer = CreditScorer(model_dir="models")


# Request/Response Models
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ApplicantFeatures(BaseModel):
    """Schema for credit application features - Auto-generated from constants."""
    
    class Config:
        extra = 'allow'  # Allow additional fields
    
    def __init__(self, **data):
        # Validate required features
        for req_feature in REQUIRED_FEATURES:
            if req_feature not in data or data[req_feature] is None:
                raise ValueError(f"Required feature missing: {req_feature}")
        
        # Validate categorical values
        for cat_feature, valid_vals in VALID_CATEGORICAL_VALUES.items():
            if cat_feature in data and data[cat_feature] is not None:
                if str(data[cat_feature]) not in [str(v) for v in valid_vals]:
                    raise ValueError(f"{cat_feature} must be one of {valid_vals}")
        
        super().__init__(**data)


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
        "version": MODEL_VERSION,
        "model": MODEL_TYPE,
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
        "approval_threshold": scorer.approval_threshold,
        "model_type": MODEL_TYPE,
        "calibration": CALIBRATION_METHOD
    }


@app.post("/score", response_model=ScoringResponse)
def score_application(applicant: ApplicantFeatures):
    """
    Score a single credit application.
    
    Returns credit decision, default probability, and confidence score.
    """
    try:
        # Convert Pydantic model to dict
        applicant_data = applicant.dict()
        
        # Get scoring result
        result = scorer.score_applicant(applicant_data, return_details=False)
        
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
        results = scorer.score_applicant(applicants_data, return_details=False)
        
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
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "features_count": len(FINAL_FEATURE_SET),
        "feature_names": FINAL_FEATURE_SET,
        "required_features": REQUIRED_FEATURES,
        "approval_threshold": scorer.approval_threshold,
        "calibration": CALIBRATION_METHOD
    }


@app.get("/features")
def get_features():
    """Get complete feature list and requirements."""
    return {
        "total_features": len(FINAL_FEATURE_SET),
        "required_features": REQUIRED_FEATURES,
        "all_features": FINAL_FEATURE_SET,
        "categorical_features": list(VALID_CATEGORICAL_VALUES.keys()),
        "feature_constraints": FEATURE_CONSTRAINTS,
        "valid_values": VALID_CATEGORICAL_VALUES
    }


if __name__ == "__main__":
    # Run with: python -m src.score_pd.api
    uvicorn.run(app, host="0.0.0.0", port=8000)