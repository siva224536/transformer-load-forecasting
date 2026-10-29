from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from features import extract_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_leakage_current.csv"
MODELS = ROOT / "models"
st.set_page_config(page_title="Insulation Degradation Classifier", layout="wide")
st.title("AI-Based Electrical Insulation Degradation Classification")
st.caption("Educational prototype using simulated leakage-current signals — not a certified diagnostic instrument.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

df = load_data()
st.sidebar.header("Prediction settings")
model_name = st.sidebar.selectbox("Classifier", ["random_forest", "hist_gradient_boosting"])
st.sidebar.info("Train models first: python src/train_model.py")
if not (MODELS / f"{model_name}.joblib").exists():
    st.warning("Trained model not found. Run `python src/train_model.py` from the project folder, then refresh.")
    st.stop()

model = joblib.load(MODELS / f"{model_name}.joblib")
columns = joblib.load(MODELS / "feature_columns.joblib")
labels = sorted(df["label"].unique().tolist())

st.subheader("Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Signal windows", len(df))
c2.metric("Samples per window", len(json.loads(df.iloc[0]["signal"])))
c3.metric("Sampling rate", f"{int(df.iloc[0]['sampling_rate_hz'])} Hz")
counts = df["label"].value_counts().reindex(labels)
st.bar_chart(counts)

st.subheader("Explore a signal")
choice = st.selectbox("Select a held-out-style demonstration record", df["sample_id"].tolist(), index=0)
row = df.loc[df["sample_id"] == choice].iloc[0]
signal = np.asarray(json.loads(row["signal"]), dtype=float)
fs = int(row["sampling_rate_hz"])
features = extract_features(signal, fs)
feature_df = pd.DataFrame([features])[columns]
prediction = model.predict(feature_df)[0]
probabilities = None
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(feature_df)[0]

left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(len(signal))/fs, signal, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Leakage current (simulated units)")
    ax.set_title("Leakage-current waveform")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
with right:
    spec = np.abs(np.fft.rfft(signal - np.mean(signal)))
    freq = np.fft.rfftfreq(len(signal), d=1/fs)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freq, spec, linewidth=0.8)
    ax.set_xlim(0, min(fs/2, 250))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("FFT magnitude")
    ax.set_title("Frequency spectrum")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)

st.subheader("Prediction")
p1, p2 = st.columns([1, 2])
p1.metric("Predicted condition", prediction)
p2.write(f"Dataset label for this demonstration record: **{row['label']}**")
if probabilities is not None:
    prob_df = pd.DataFrame({"Class": model.classes_, "Probability": probabilities}).set_index("Class")
    st.bar_chart(prob_df)
st.subheader("Extracted features")
st.dataframe(pd.DataFrame([features]).T.rename(columns={0:"Value"}), use_container_width=True)
st.caption("A prediction on a randomly selected record from the provided dataset is for interface demonstration only. For valid evaluation, use the saved unseen test metrics, and preferably an external dataset/device.")
