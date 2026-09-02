"""
Step 4: Monitoring dashboard.

Reads predictions.db (populated by app.py) and visualizes what the model
has been predicting over time. This is a toy version of what a real
monitoring system (Evidently, Grafana + Prometheus, an in-house dashboard)
does: watch the live prediction distribution for signs it's drifting away
from what the model saw during training.

Run:
    streamlit run dashboard.py

(Needs predictions.db to have some rows in it -- hit the /predict endpoint
a few times first, or run simulate_traffic.py.)
"""
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Monitoring", layout="wide")
st.title("Toy Model Monitoring Dashboard")

DB_PATH = Path("predictions.db")

if not DB_PATH.exists():
    st.warning(
        "No predictions.db found yet. Start the API (`uvicorn app:app --reload`) "
        "and send it a few requests, or run `python simulate_traffic.py`."
    )
    st.stop()

con = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp", con)
con.close()

if df.empty:
    st.info("predictions.db exists but is empty -- send some requests first.")
    st.stop()

df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

col1, col2, col3 = st.columns(3)
col1.metric("Total predictions logged", len(df))
col2.metric("Mean predicted probability", f"{df['prediction'].mean():.3f}")
col3.metric("p95 latency (ms)", f"{df['latency_ms'].quantile(0.95):.1f}")

st.subheader("Prediction distribution over time")
st.caption(
    "This is the core drift-detection signal: if this shape shifts meaningfully "
    "from what you saw during training/validation, something about the incoming "
    "data has changed -- new fraud patterns, a broken upstream feature, a new "
    "user segment, etc. -- and it's worth investigating before trusting the model."
)
st.line_chart(df.set_index("datetime")["prediction"])

st.subheader("Prediction value histogram")
st.bar_chart(df["prediction"].value_counts(bins=20).sort_index())

st.subheader("Latency over time")
st.line_chart(df.set_index("datetime")["latency_ms"])

st.subheader("Raw prediction log (most recent 50)")
st.dataframe(df.sort_values("timestamp", ascending=False).head(50))
