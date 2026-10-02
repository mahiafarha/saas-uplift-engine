import pandas as pd
import numpy as np

# Load our generated data
df = pd.read_csv('saas_churn_experiment.csv')

print("=" * 60)
print("1. SAMPLE RATIO MISMATCH (SRM) CHECK")
print("=" * 60)
total_users = len(df)
treatment_count = df['treatment'].sum()
control_count = total_users - treatment_count
print(f"Total Users: {total_users}")
print(f"Control (0): {control_count} ({control_count / total_users:.1%})")
print(f"Treatment (1): {treatment_count} ({treatment_count / total_users:.1%})")

print("\n" + "=" * 60)
print("2. COVARIATE BALANCE (Pre-Treatment Equivalence)")
print("=" * 60)
# The treatment and control groups must look identical on pre-treatment metrics
balance_features = ['monthly_fee', 'tenure_days', 'logins_last_30d', 'usage_drop_ratio', 'support_tickets']
balance_table = df.groupby('treatment')[balance_features].mean().T
balance_table.columns = ['Control Mean', 'Treatment Mean']
balance_table['Difference'] = balance_table['Treatment Mean'] - balance_table['Control Mean']
print(balance_table.round(3))

print("\n" + "=" * 60)
print("3. OVERALL EXPERIMENT RESULTS (Naïve A/B Test Outcome)")
print("=" * 60)
control_churn = df[df['treatment'] == 0]['churn'].mean()
treatment_churn = df[df['treatment'] == 1]['churn'].mean()
average_treatment_effect = treatment_churn - control_churn

print(f"Control Churn Rate:   {control_churn:.2%}")
print(f"Treatment Churn Rate: {treatment_churn:.2%}")
print(f"Average Treatment Effect (ATE): {average_treatment_effect:+.2%} points")

print("\n" + "=" * 60)
print("4. HETEROGENEOUS TREATMENT EFFECT (The Core Insight)")
print("=" * 60)
# Group users into 3 activity health buckets based on usage drop ratio
df['usage_segment'] = pd.cut(
    df['usage_drop_ratio'],
    bins=[0, 0.6, 1.2, 3.0],
    labels=['Declining Usage', 'Stable Usage', 'Surging Usage']
)

segment_analysis = df.groupby(['usage_segment', 'treatment'])['churn'].mean().unstack()
segment_analysis.columns = ['Control Churn', 'Treatment Churn']
segment_analysis['Uplift (Drop in Churn)'] = segment_analysis['Treatment Churn'] - segment_analysis['Control Churn']
print((segment_analysis * 100).round(2).astype(str) + '%')