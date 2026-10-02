from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Load trained causal artifacts
artifacts = joblib.load('uplift_tlearner_artifacts.joblib')
m0 = artifacts['model_control']
m1 = artifacts['model_treatment']
tier_map = artifacts['tier_map']
cost_per_contact = artifacts['cost_per_contact']

app = FastAPI(
    title="SaaS Causal Uplift & Prescriptive Intervention API",
    version="1.0.0",
    description="Scores individual treatment effects (ITE) and issues prescriptive outreach recommendations."
)

class CustomerPayload(BaseModel):
    user_id: str = Field(..., example="usr_90412")
    plan_tier: str = Field(..., example="Professional")
    monthly_fee: float = Field(..., example=99.0)
    tenure_days: int = Field(..., example=140)
    logins_last_30d: int = Field(..., example=12)
    usage_drop_ratio: float = Field(..., example=0.45)
    support_tickets: int = Field(..., example=3)

class PrescriptiveResponse(BaseModel):
    user_id: str
    baseline_churn_risk: float
    treatment_churn_risk: float
    predicted_uplift: float
    expected_roi_dollars: float
    recommendation: str
    action_reasoning: str

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict-intervention", response_model=PrescriptiveResponse)
def evaluate_customer(payload: CustomerPayload):
    tier_code = tier_map.get(payload.plan_tier)
    if tier_code is None:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid plan tier. Choose from: {list(tier_map.keys())}"
        )

    vector = np.array([[
        tier_code,
        payload.monthly_fee,
        payload.tenure_days,
        payload.logins_last_30d,
        payload.usage_drop_ratio,
        payload.support_tickets
    ]])

    # Counterfactual predictions: P(churn | T=0) vs P(churn | T=1)
    p_ctrl = float(m0.predict_proba(vector)[0, 1])
    p_trt = float(m1.predict_proba(vector)[0, 1])
    uplift = p_ctrl - p_trt  # Positive = churn reduction

    # 6-month CLV retention estimation
    intervention_value = 6 * payload.monthly_fee
    expected_roi = (uplift * intervention_value) - cost_per_contact

    # Prescriptive decision rule
    if expected_roi > 0 and uplift > 0.05:
        recommendation = "TARGET_WITH_OFFER"
        reasoning = (
            f"Persuadable profile: Outreach predicted to decrease churn by {uplift:.1%} "
            f"with expected net return of ${expected_roi:.2f}."
        )
    elif uplift <= -0.02:
        recommendation = "DO_NOT_DISTURB"
        reasoning = (
            f"Potential sleeping dog: Outreach predicted to accelerate churn by {abs(uplift):.1%}."
        )
    else:
        recommendation = "PASSIVE_MONITORING"
        reasoning = (
            f"Insufficient uplift ({uplift:.1%}) to clear intervention hurdle cost of ${cost_per_contact:.2f}."
        )

    return PrescriptiveResponse(
        user_id=payload.user_id,
        baseline_churn_risk=round(p_ctrl, 4),
        treatment_churn_risk=round(p_trt, 4),
        predicted_uplift=round(uplift, 4),
        expected_roi_dollars=round(expected_roi, 2),
        recommendation=recommendation,
        action_reasoning=reasoning
    )