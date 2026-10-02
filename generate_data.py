import numpy as np
import pandas as pd

# Set seed for reproducible numbers
np.random.seed(42)
n_users = 10_000

# 1. Base User Features
tenure_days = np.random.exponential(scale=200, size=n_users).astype(int) + 15
plan_tiers = np.random.choice(['Starter', 'Professional', 'Enterprise'], size=n_users, p=[0.6, 0.3, 0.1])
fee_map = {'Starter': 29.0, 'Professional': 99.0, 'Enterprise': 299.0}
monthly_fee = np.array([fee_map[tier] for tier in plan_tiers])

# 2. Behavioral Signals
logins_last_30d = np.random.poisson(lam=25, size=n_users)
# Usage drop ratio: < 1.0 means usage is slowing down recently
usage_drop_ratio = np.random.normal(loc=0.9, scale=0.3, size=n_users).clip(0.1, 2.5)
support_tickets = np.random.poisson(lam=1.5, size=n_users)

# 3. Randomized A/B Experiment Assignment (50/50 split)
# treatment = 1 means customer received proactive outreach/discount
treatment = np.random.binomial(n=1, p=0.5, size=n_users)

# 4. Underlying Latent Churn Propensity
# More tickets, dropping usage, and high fees push churn risk UP. Long tenure pushes it DOWN.
latent_risk = (
    0.4 * (support_tickets > 2)
    + 0.5 * (usage_drop_ratio < 0.7)
    + 0.2 * (monthly_fee / 100.0)
    - 0.3 * (tenure_days > 180)
    + np.random.normal(0, 0.2, size=n_users)
)

# Convert latent risk into baseline probability of churn (between 0 and 1)
baseline_churn_prob = 1 / (1 + np.exp(-latent_risk))

# 5. Causal Effect (Treatment Uplift)
# The offer cuts churn probability significantly for users with dropping usage (Persuadables)
# For disengaged/inactive users, outreach might remind them to cancel (Sleeping Dogs)
treatment_effect = np.where(
    usage_drop_ratio < 0.6,
    -0.25, # Persuadables: Offer cuts churn probability by 25%
    np.where(usage_drop_ratio > 1.4, 0.05, -0.05) # Sleeping dogs vs minor help
)

# Final probability of churn depending on whether they got treatment
final_churn_prob = np.clip(baseline_churn_prob + treatment * treatment_effect, 0.01, 0.99)
churn = np.random.binomial(n=1, p=final_churn_prob, size=n_users)

# Combine into a clean DataFrame
df = pd.DataFrame({
    'user_id': [f"usr_{i:05d}" for i in range(n_users)],
    'plan_tier': plan_tiers,
    'monthly_fee': monthly_fee,
    'tenure_days': tenure_days,
    'logins_last_30d': logins_last_30d,
    'usage_drop_ratio': np.round(usage_drop_ratio, 2),
    'support_tickets': support_tickets,
    'treatment': treatment,
    'churn': churn
})

# Save to CSV
df.to_csv('saas_churn_experiment.csv', index=False)
print(f"Generated {len(df)} customer records successfully!")
print("\nSample records:")
print(df.head())