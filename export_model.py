import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Load training dataset
df = pd.read_csv('saas_churn_experiment.csv')
tier_map = {'Starter': 1, 'Professional': 2, 'Enterprise': 3}
df['plan_tier_code'] = df['plan_tier'].map(tier_map)

feature_cols = [
    'plan_tier_code', 'monthly_fee', 'tenure_days',
    'logins_last_30d', 'usage_drop_ratio', 'support_tickets'
]

# Separate control and treatment
ctrl = df[df['treatment'] == 0]
trt = df[df['treatment'] == 1]

# Train production T-Learner estimators
m0 = HistGradientBoostingClassifier(random_state=42, max_iter=100)
m0.fit(ctrl[feature_cols], ctrl['churn'])

m1 = HistGradientBoostingClassifier(random_state=42, max_iter=100)
m1.fit(trt[feature_cols], trt['churn'])

# Bundle pipeline metadata
artifacts = {
    'model_control': m0,
    'model_treatment': m1,
    'features': feature_cols,
    'tier_map': tier_map,
    'cost_per_contact': 25.0
}

joblib.dump(artifacts, 'uplift_tlearner_artifacts.joblib')
print("Model pipeline serialized to 'uplift_tlearner_artifacts.joblib'")