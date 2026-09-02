# Toy MLOps Pipeline

A minimal but *complete* loop through the tools real ML teams use, applied to
a toy binary classification problem (sklearn's breast cancer dataset —
30 numeric features, binary label — deliberately shaped like a fraud problem:
tabular, binary, imbalance-adjacent).

The point isn't the model. It's everything around it. Once this clicks,
swapping in your fraud dataset later is just replacing `train.py`'s data
loading and `app.py`'s `PredictRequest` schema — the rest of the stack
(Docker, serving, logging, monitoring, CI) barely changes.

## What's here

| File | Stage | Tool |
|---|---|---|
| `train.py` | Train + track experiments | MLflow |
| `app.py` | Serve predictions over HTTP | FastAPI |
| `Dockerfile` | Package it portably | Docker |
| `dashboard.py` | Watch live predictions for drift | Streamlit |
| `simulate_traffic.py` | Generate fake traffic + inject drift | httpx |
| `test_app.py` | Catch breakage before deploy | pytest |
| `.github/workflows/ci.yml` | Auto-run tests on every push | GitHub Actions |

## Step 1 — Train, with experiment tracking

```bash
pip install -r requirements.txt
python train.py
```

This trains a `RandomForestClassifier`, logs params/metrics/the model itself
to MLflow, and writes `model.pkl` + `feature_names.json` to disk (what the
API actually loads).

Look at the tracking dashboard:
```bash
mlflow ui
```
Open `http://localhost:5000`. Every run you do (change `n_estimators`, rerun)
shows up here — this is what replaces "I think that run with dropout 0.3 was
better, let me check my print statements" with an actual comparable history.

## Step 2 — Serve it

```bash
uvicorn app:app --reload
```

Test it:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
```

Every call gets logged to `predictions.db` (SQLite) — timestamp, input,
output, latency. This is the seed of the monitoring step below; you can't
detect drift in data you never recorded.

`/health` is also there — deployment platforms poll this to know your
container is alive before routing real traffic to it.

## Step 3 — Containerize it

```bash
docker build -t toy-mlops-api .
docker run -p 8000:8000 toy-mlops-api
```

Same `curl` command as above should work identically — that's the whole
point of the container. It behaves the same on your laptop, a teammate's
laptop, or a cloud VM, because it's not relying on *your* Python install.

## Step 4 — Deploy it somewhere real

Easiest path for a portfolio project: **Render** or **Railway**, free tier.
Both let you connect a GitHub repo and auto-build from your `Dockerfile` —
push to `main`, it deploys. No manual server setup.

1. Push this repo to GitHub
2. On Render: New → Web Service → connect the repo → it detects the
   Dockerfile automatically → deploy
3. You'll get a public URL — hit `/health` on it to confirm it's live

(AWS/GCP are more "industry-standard" but have a lot more setup overhead —
save those for once this pattern is comfortable.)

## Step 5 — Monitor it

With the API running, generate some traffic (including deliberately
injected drift):
```bash
python simulate_traffic.py
```

Then look at it:
```bash
streamlit run dashboard.py
```

You should see the prediction distribution visibly shift partway through —
that's the moment `simulate_traffic.py` started sending artificially scaled
("drifted") inputs. In a real system, a shift like this is your signal that
something changed: new fraud patterns, a broken upstream feature, a new
user segment, a currency-unit bug, etc. This is the piece most portfolio
projects skip entirely, and it's the part that actually demonstrates you
understand *production* ML, not just modeling.

## Step 6 — CI

`.github/workflows/ci.yml` runs on every push: retrains the model fresh
(so tests never run against a stale artifact), runs `pytest`, and builds
the Docker image. Push to GitHub and check the Actions tab — a red X means
something broke before it ever reached deployment.

## Applying this to the fraud project

When you're ready to move to the real project, the changes are:
- `train.py`: swap the sklearn toy dataset for your fraud data + your
  supervised/unsupervised/PU-learning blend
- `app.py`: swap `PredictRequest` for named transaction fields (amount,
  merchant category, etc.) instead of a raw float list — worth doing for
  real, since named fields catch schema mistakes early
- `dashboard.py`: the drift story becomes much more real — you can split
  by time period using actual transaction timestamps rather than a
  synthetic injection
- Everything else (Docker, FastAPI structure, CI, SQLite logging) carries
  over almost unchanged
