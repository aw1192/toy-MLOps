"""
Optional: generates fake requests against the running API so the dashboard
has something to show, and deliberately injects drift partway through so you
can SEE the dashboard catch it -- this is the payoff of the whole monitoring
setup, and the one piece most portfolio projects skip.

Run this while `uvicorn app:app --reload` is running in another terminal:
    python simulate_traffic.py
"""
import time

import httpx
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data

API_URL = "http://localhost:8000/predict"
N_NORMAL = 40
N_DRIFTED = 40

print(f"Sending {N_NORMAL} normal requests (sampled from real training-like data)...")
for _ in range(N_NORMAL):
    row = X[np.random.randint(len(X))].tolist()
    r = httpx.post(API_URL, json={"features": row}, timeout=10)
    print(r.json())
    time.sleep(0.05)

print(f"\nNow injecting {N_DRIFTED} DRIFTED requests (features scaled way up, "
      "simulating e.g. a currency unit bug or a genuinely new transaction pattern)...")
for _ in range(N_DRIFTED):
    row = (X[np.random.randint(len(X))] * 3.0).tolist()  # artificial shift
    r = httpx.post(API_URL, json={"features": row}, timeout=10)
    print(r.json())
    time.sleep(0.05)

print("\nDone. Run `streamlit run dashboard.py` and look at the prediction "
      "distribution over time -- you should see a visible shift partway through.")
