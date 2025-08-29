import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from random import randint

def generate_synthetic_loans(n=5000, random_state=42):
    rng = np.random.default_rng(random_state)
    
    # Features
    credit_score = rng.integers(500, 851, size=n)
    annual_income = rng.integers(50, 401, size=n) * 1000  # multiples of 1000
    loan_amount = rng.integers(1, 31, size=n) * 5000  # 5k to 150k
    loan_term_months = rng.integers(1, 11, size=n) * 12  # 12 to 120
    race = rng.choice(["GroupA", "GroupB", "GroupC", "GroupD"], size=n, p=[0.25, 0.2, 0.05, 0.50])
    
    # Base probability of default
    base_prob = 0.08  # ~8% baseline
    
    # Default probability with sensible weights
    prob_default = (
        base_prob
        - 0.00015 * (credit_score - 650)   # higher credit_score → lower prob
        - 0.0000005 * (annual_income - 100000)  # higher income → lower prob
        + 0.000002 * (loan_amount - 50000)  # bigger loan → higher prob
        - 0.0005 * (loan_term_months - 60)  # longer term → lower prob
    )
    
    # Clip probs into [0.01, 0.5] to avoid extremes
    prob_default = np.clip(prob_default, 0.01, 0.5)
    
    # Draw outcomes
    loan_status = rng.binomial(1, prob_default)
    
    # Ensure <15% defaults
    current_default_rate = loan_status.mean()
    if current_default_rate > 0.15:
        # Downsample defaults to ~15%
        idx_default = np.where(loan_status == 1)[0]
        keep_defaults = rng.choice(idx_default, size=int(0.15*n), replace=False)
        keep_non_defaults = np.where(loan_status == 0)[0]
        keep_idx = np.concatenate([keep_defaults, keep_non_defaults])
        
        credit_score = credit_score[keep_idx]
        annual_income = annual_income[keep_idx]
        loan_amount = loan_amount[keep_idx]
        loan_term_months = loan_term_months[keep_idx]
        race = race[keep_idx]
        loan_status = loan_status[keep_idx]
    
    # Assemble into DataFrame
    df = pd.DataFrame({
        "credit_score": credit_score,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "race": race,
        "loan_status": loan_status
    })
    # if the annual salary is less than thrice the annual loan amount, it should be an automatic loan default 
    yearly_repayment = df["loan_amount"] / (df["loan_term_months"] / 12)
    df.loc[df["annual_income"] < 3 * yearly_repayment, "loan_status"] = 1
    
    return df

# Example usage
df = generate_synthetic_loans(n=10000, random_state=randint(1,50))
df.to_csv('./data/raw/mock_loan_data.csv', index=False)
print(df.head())
print("Default rate:", df['loan_status'].mean())

