import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('saas_churn_experiment.csv')

# Encode plan_tier as numeric categories
tier_mapping = {'Starter': 1, 'Professional': 2, 'Enterprise': 3}
df['plan_tier_code'] = df['plan_tier'].map(tier_mapping)

feature_cols = [
    'plan_tier_code', 'monthly_fee', 'tenure_days',
    'logins_last_30d', 'usage_drop_ratio', 'support_tickets'
]

# 2. Split into Train & Evaluation sets (80 / 20)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

# Separate into Control and Treatment sub-datasets within training
control_train = train_df[train_df['treatment'] == 0]
treatment_train = train_df[train_df['treatment'] == 1]

# 3. Train Model 0 (Control) and Model 1 (Treatment)
print("Training Control Model (M0)...")
m0 = HistGradientBoostingClassifier(random_state=42, max_iter=100)
m0.fit(control_train[feature_cols], control_train['churn'])

print("Training Treatment Model (M1)...")
m1 = HistGradientBoostingClassifier(random_state=42, max_iter=100)
m1.fit(treatment_train[feature_cols], treatment_train['churn'])

# 4. Predict on Test Set
# We predict churn probability under Control vs Treatment for the unseen test users
test_features = test_df[feature_cols]

# Predict probability of churn (class 1)
prob_churn_if_control = m0.predict_proba(test_features)[:, 1]
prob_churn_if_treatment = m1.predict_proba(test_features)[:, 1]

# Uplift = (Churn Risk without Offer) - (Churn Risk with Offer)
# Positive uplift = The offer saves the customer from leaving
test_df = test_df.copy()
test_df['predicted_uplift'] = prob_churn_if_control - prob_churn_if_treatment

print("\n" + "=" * 60)
print("T-LEARNER PREDICTION SUMMARY (Unseen Test Set)")
print("=" * 60)
print(test_df[['user_id', 'usage_drop_ratio', 'predicted_uplift']].head(10))

# 5. Segment into Deciles to verify ranking
test_df['uplift_decile'] = pd.qcut(test_df['predicted_uplift'], q=5, labels=['Lowest', 'Low', 'Medium', 'High', 'Highest'])
decile_summary = test_df.groupby('uplift_decile')[['predicted_uplift', 'usage_drop_ratio']].mean()
print("\nMean Predicted Uplift by Quintile:")
print(decile_summary.round(3))

# Save scored test set for financial evaluation
test_df.to_csv('test_scored_with_uplift.csv', index=False)
print("\nSaved predictions to 'test_scored_with_uplift.csv'")