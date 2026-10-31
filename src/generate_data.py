
import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible sample data
np.random.seed(42)

N = 5000

amount = np.random.lognormal(mean=4, sigma=1.2, size=N).round(2)
hour = np.random.randint(0, 24, size=N)
distance = np.random.exponential(scale=20, size=N).round(2)
failed_attempts = np.random.poisson(lam=0.3, size=N)
international = np.random.binomial(1, 0.15, size=N)

# Create a synthetic fraud label for demonstration
logit = (
    -4.0
    + 0.025 * np.minimum(amount, 200)
    + 0.8 * (hour <= 4)
    + 0.025 * np.minimum(distance, 50)
    + 0.8 * failed_attempts
    + 1.0 * international
)

probability = 1 / (1 + np.exp(-logit))
is_fraud = np.random.binomial(1, probability)

df = pd.DataFrame({
    "transaction_id": range(1, N + 1),
    "amount": amount,
    "hour": hour,
    "distance_from_usual_km": distance,
    "failed_attempts": failed_attempts,
    "international": international,
    "is_fraud": is_fraud,
})

output_path = Path("data/transactions.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")
print(f"Total transactions: {len(df)}")
print(f"Fraudulent transactions: {df['is_fraud'].sum()}")
print(f"Legitimate transactions: {(df['is_fraud'] == 0).sum()}")
print("\nFirst five transactions:")
print(df.head())
