import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, runpy, json, io
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
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    st.download_button("📥 Download Dataset (CSV)", df.to_csv(index=False).encode("utf-8"), "dataset.csv", "text/csv")

    st.write("### Statistical Summary")
    st.dataframe(df.describe().T)
    st.download_button("📥 Download Summary (CSV)", df.describe().to_csv().encode("utf-8"), "summary.csv", "text/csv")

    st.write("### Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="status", data=df, palette="Set2", ax=ax)
    st.pyplot(fig)
    st.download_button("📥 Download Target Distribution (PNG)", export_fig(fig).getvalue(), "target_distribution.png", "image/png")

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
    st.download_button("📥 Download Heatmap (PNG)", export_fig(fig).getvalue(), "heatmap.png", "image/png")

    st.write("### Feature Distributions")
    num_cols = df.select_dtypes(include=np.number).columns
    fig, axes = plt.subplots(len(num_cols)//3+1, 3, figsize=(15, len(num_cols)*2))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="skyblue")
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)
    st.download_button("📥 Download Histograms (PNG)", export_fig(fig).getvalue(), "histograms.png", "image/png")

    st.write("### Boxplots by Target")
    fig, axes = plt.subplots(len(num_cols)//3+1, 3, figsize=(15, len(num_cols)*2))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(x="status", y=col, data=df, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)
    st.download_button("📥 Download Boxplots (PNG)", export_fig(fig).getvalue(), "boxplots.png", "image/png")

    st.write("### PCA Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", alpha=0.7)
    ax.set_title("PCA Projection")
    st.pyplot(fig)
    st.download_button("📥 Download PCA (PNG)", export_fig(fig).getvalue(), "pca.png", "image/png")

    st.write("### t-SNE Visualization (sampled, may take time)")
    sample_size = min(300, len(X))  # limit to 300 samples for speed
    X_sample = X.sample(sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    X_tsne = tsne.fit_transform(X_sample)
    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y_sample, cmap="coolwarm", alpha=0.7)
    ax.set_title("t-SNE Projection (sample of 300)")
    st.pyplot(fig)
    st.download_button("📥 Download t-SNE (PNG)", export_fig(fig).getvalue(), "tsne.png", "image/png")

    st.write("### Feature Importance (if available)")
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x=importances, y=importances.index, ax=ax)
        st.pyplot(fig)
        st.download_button("📥 Download Feature Importance (PNG)", export_fig(fig).getvalue(), "feature_importance.png", "image/png")

# (Tabs 2–4 stay the same as in v20)
