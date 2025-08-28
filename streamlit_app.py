import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, runpy, json, io, shutil
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

st.set_page_config(page_title="Parkinson’s ML App", page_icon="🧠", layout="wide")

# ==============================
# Helpers
# ==============================
def safe_predict(model, X):
    try:
        return model.predict(X)
    except ValueError:
        if hasattr(model, "feature_names_in_"):
            X = X[model.feature_names_in_]
        return model.predict(X)

def safe_predict_proba(model, X):
    try:
        return model.predict_proba(X)
    except ValueError:
        if hasattr(model, "feature_names_in_"):
            X = X[model.feature_names_in_]
        return model.predict_proba(X)

def risk_label(prob, threshold=0.5):
    if prob < 0.3:
        return "🟢 Low"
    elif prob < 0.7:
        return "🟡 Medium"
    else:
        return "🔴 High"

def decision_text(prob, threshold=0.5):
    decision = "Positive (Parkinson’s)" if prob >= threshold else "Negative (Healthy)"
    return f"ההסתברות היא {prob*100:.1f}%, הסיווג עם הסף {threshold:.2f} הוא {decision}"

def export_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf

# ==============================
# Load dataset
# ==============================
DATA_PATH = "data/parkinsons.csv"
df = pd.read_csv(DATA_PATH)
if "name" in df.columns:
    df = df.drop(columns=["name"])
X = df.drop("status", axis=1)
y = df["status"]

# ==============================
# Load model + metrics once
# ==============================
def load_model_and_metrics():
    if not os.path.exists("models/best_model.joblib") or not os.path.exists("assets/metrics.json"):
        runpy.run_path("model_pipeline.py")
    best_model = joblib.load("models/best_model.joblib")
    with open("assets/metrics.json","r") as f:
        metrics = json.load(f)
    return best_model, metrics

if "best_model" not in st.session_state or "metrics" not in st.session_state:
    st.session_state.best_model, st.session_state.metrics = load_model_and_metrics()

best_model = st.session_state.best_model
metrics = st.session_state.metrics

# ==============================
# Tabs
# ==============================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data & EDA", 
    "🤖 Models", 
    "🔮 Prediction", 
    "⚡ Train New Model"
])

# --- Tab 1: Data & EDA
with tab1:
    st.header("📊 Data & Exploratory Data Analysis")
    st.dataframe(df.head())

# --- Tab 2: Models
with tab2:
    st.header("🤖 Model Training & Comparison")
    df_metrics = pd.DataFrame(metrics.items(), columns=["Model","ROC-AUC"]).sort_values(by="ROC-AUC", ascending=False)
    st.dataframe(df_metrics)
    best_name = df_metrics.iloc[0]["Model"]
    st.success(f"🏆 Best Model: {best_name} (ROC-AUC={df_metrics.iloc[0]['ROC-AUC']:.3f})")

# --- Tab 3: Prediction
with tab3:
    st.header("🔮 Prediction")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
    option = st.radio("Choose input type:", ["Manual Input","Upload CSV"])
    
    if option=="Manual Input":
        inputs = {col: st.number_input(col, float(X[col].mean())) for col in X.columns}
        sample = pd.DataFrame([inputs])
        if st.button("Predict Sample"):
            prob = safe_predict_proba(best_model, sample)[0,1]
            st.progress(prob)
            st.write(risk_label(prob, threshold))
            st.info(decision_text(prob, threshold))
    
    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            new_df = pd.read_csv(file)
            probs = safe_predict_proba(best_model, new_df)[:,1]
            preds = (probs >= threshold).astype(int)
            new_df["Probability"] = probs
            new_df["Prediction"] = preds
            new_df["risk_label"] = [risk_label(p, threshold) for p in probs]
            new_df["decision_text"] = [decision_text(p, threshold) for p in probs]
            st.dataframe(new_df.head())

# --- Tab 4: Train New Model
with tab4:
    st.header("⚡ Train New Model")
    file = st.file_uploader("Upload CSV for retraining", type=["csv"], key="newtrain")
    if file:
        new_df = pd.read_csv(file)
        st.write("New Data Preview:", new_df.head())
        
        if st.button("Retrain Models"):
            # Save new training data
            new_path = "data/new_train.csv"
            new_df.to_csv(new_path, index=False)
            # Run pipeline (expected to save models/best_model_new.joblib and assets/metrics_new.json)
            runpy.run_path("model_pipeline.py")

            # Load current + new metrics
            old_metrics = metrics
            if os.path.exists("assets/metrics_new.json"):
                with open("assets/metrics_new.json","r") as f:
                    new_metrics = json.load(f)
            else:
                new_metrics = {"NewModel":0.0}

            old_auc = max(old_metrics.values())
            new_auc = max(new_metrics.values())

            st.subheader("📊 Comparison of Best Models")
            comp_df = pd.DataFrame({
                "Model": ["Current Best","New Best"],
                "ROC-AUC": [old_auc, new_auc]
            })
            st.dataframe(comp_df)

            # Store paths in session for optional promotion
            st.session_state.new_best_model = "models/best_model_new.joblib"
            st.session_state.new_metrics = new_metrics

        if "new_best_model" in st.session_state:
            if st.button("🚀 Promote New Model"):
                # Replace current with new
                shutil.copy(st.session_state.new_best_model, "models/best_model.joblib")
                with open("assets/metrics.json","w") as f:
                    json.dump(st.session_state.new_metrics, f)
                st.success("✅ New model promoted and is now active!")
                st.rerun()
