import streamlit as st
import pandas as pd
import joblib
import json
import os
import runpy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

st.set_page_config(page_title="Parkinson’s Predictor", page_icon="🧠")

# ==============================
# Load dataset for feature names
# ==============================
DATA_PATH = "data/parkinsons.csv"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    # download dataset if not available locally
    df = pd.read_csv(UCI_URL)
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

X = df.drop("status", axis=1)
y = df["status"]

# ==============================
# Helper to load model + metrics
# ==============================
def load_model_and_metrics():
    # ensure directories exist
    os.makedirs("assets", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # run pipeline if files are missing
    if not os.path.exists("models/best_model.joblib") or not os.path.exists("assets/metrics.json"):
        runpy.run_path("model_pipeline.py")

    # if metrics still missing, create default placeholder
    if not os.path.exists("assets/metrics.json"):
        with open("assets/metrics.json", "w") as f:
            json.dump({"Info": "No metrics available"}, f)

    # if model still missing, raise clear error
    if not os.path.exists("models/best_model.joblib"):
        raise FileNotFoundError("best_model.joblib was not created by model_pipeline.py")

    best_model = joblib.load("models/best_model.joblib")
    with open("assets/metrics.json", "r") as f:
        metrics = json.load(f)
    return best_model, metrics

# Initialize session_state
if "best_model" not in st.session_state or "metrics" not in st.session_state:
    st.session_state.best_model, st.session_state.metrics = load_model_and_metrics()

best_model = st.session_state.best_model
metrics = st.session_state.metrics

st.title("🧠 Parkinson’s Disease Prediction App")
st.write("This app predicts the likelihood of Parkinson’s Disease based on voice features.")

# ==============================
# Tabs for navigation
# ==============================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "📈 Performance Charts", "🧩 Confusion Matrix", "📉 ROC Curve"])

# --- Tab 1: Model Comparison (Table) ---
with tab1:
    st.header("📊 Model Comparison (Table)")
    df_metrics = pd.DataFrame(metrics.items(), columns=["Model", "ROC-AUC"])
    st.dataframe(df_metrics)

# --- Tab 2: Performance Charts ---
with tab2:
    st.header("📈 Model Comparison (Bar Chart)")
    st.bar_chart(df_metrics.set_index("Model"))

# --- Tab 3: Confusion Matrix ---
with tab3:
    st.header("🧩 Confusion Matrix (Best Model)")
    y_pred = best_model.predict(X)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Parkinson’s"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    st.pyplot(fig)

# --- Tab 4: ROC Curve ---
with tab4:
    st.header("📉 ROC Curve (Best Model)")
    y_pred_prob = best_model.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)

# ==============================
# Prediction Section
# ==============================
st.header("🔍 New Prediction")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))
sample = pd.DataFrame([input_data])

if st.button("Predict"):
    pred_prob = best_model.predict_proba(sample)[0,1]
    pred = int(pred_prob >= 0.5)
    if pred == 1:
        st.error(f"❌ Parkinson’s (probability: {pred_prob:.2f})")
    else:
        st.success(f"✅ Healthy (probability: {1-pred_prob:.2f})")

# ==============================
# Promote Button
# ==============================
st.header("⚡ Promote (Re-train Best Model)")
if st.button("Promote Model (Re-train)"):
    runpy.run_path("model_pipeline.py")
    st.success("✔️ Model retrained and promoted successfully!")

    # Reload after retrain
    st.session_state.best_model, st.session_state.metrics = load_model_and_metrics()
    st.experimental_rerun()
