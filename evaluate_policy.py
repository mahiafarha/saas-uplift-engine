import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Scored Test Data
df = pd.read_csv('test_scored_with_uplift.csv')

# Sort descending by predicted uplift (best treatment candidates first)
df = df.sort_values(by='predicted_uplift', ascending=False).reset_index(drop=True)

# 2. Cumulative Uplift / Qini Calculations
# For each top k% fraction of users targeted:
# Qini(k) = (Retained in Treatment) - (Retained in Control) * (N_T / N_C)
df['is_retained'] = 1 - df['churn']
df['n_t'] = (df['treatment'] == 1).astype(int)
df['n_c'] = (df['treatment'] == 0).astype(int)
df['retained_t'] = df['is_retained'] * df['n_t']
df['retained_c'] = df['is_retained'] * df['n_c']

# Cumulative sums along the ranked list
cum_n_t = df['n_t'].cumsum()
cum_n_c = df['n_c'].cumsum()
cum_ret_t = df['retained_t'].cumsum()
cum_ret_c = df['retained_c'].cumsum()

# Uplift curve metric: incremental retained users
# Handling division by zero safely at the start
scale_factor = np.where(cum_n_c == 0, 1.0, cum_n_t / cum_n_c)
qini_curve = cum_ret_t - (cum_ret_c * scale_factor)

# 3. Financial Prescriptive Simulation
# What is the actual financial gain vs contacting everyone or contacting nobody?
cost_per_contact = 25.0
intervention_value = 6 * df['monthly_fee']  # 6-month retention value

# Expected Net Benefit for each customer = (Uplift * Value) - Cost
df['expected_net_benefit'] = (df['predicted_uplift'] * intervention_value) - cost_per_contact

# Target only where expected net benefit > 0
df['should_target'] = df['expected_net_benefit'] > 0
targeted_users = df[df['should_target']]

print("=" * 65)
print("FINANCIAL POLICY PRESCRIPTIVE SUMMARY")
print("=" * 65)
print(f"Total Test Cohort:              {len(df):,} customers")
print(f"Optimal Cohort to Target:       {len(targeted_users):,} customers ({len(targeted_users)/len(df):.1%})")
print(f"Average Predicted Uplift (Targeted): {targeted_users['predicted_uplift'].mean():.2%}")
print(f"Total Projected Net Profit Lift: ${targeted_users['expected_net_benefit'].sum():,.2f}")

# Contrast with naive policy (target everyone)
naive_all_profit = df['expected_net_benefit'].sum()
print(f"Targeting Everyone (Naive) Net:  ${naive_all_profit:,.2f}")
print(f"Efficiency Advantage:            +${targeted_users['expected_net_benefit'].sum() - naive_all_profit:,.2f}")

# 4. Plot Cumulative Qini Curve
plt.figure(figsize=(9, 5))
x_axis = np.linspace(0, 100, len(df))

plt.plot(x_axis, qini_curve, label='T-Learner Policy', color='#1f77b4', lw=2.5)
# Random targeting benchmark line
plt.plot([0, 100], [0, qini_curve.iloc[-1]], label='Random Targeting Baseline', color='#d62728', linestyle='--')

plt.title('Cumulative Qini Curve (Evaluation on Unseen Test Set)', fontsize=13, pad=12)
plt.xlabel('Percentage of Customer Base Targeted (%)', fontsize=11)
plt.ylabel('Incremental Retained Customers', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('qini_curve.png', dpi=300)
print("\nPlot saved successfully as 'qini_curve.png'")
