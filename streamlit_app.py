import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, runpy, json
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

st.set_page_config(page_title="Parkinson’s ML App", page_icon="🧠", layout="wide")

# ==============================
# Safe + Fast prediction helpers
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

def fast_predict(model, df):
    return model.predict(df.to_numpy())

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
# Load model + metrics
# ==============================
def load_model_and_metrics():
    if not os.path.exists("models/best_model.joblib") or not os.path.exists("assets/metrics.json"):
        runpy.run_path("model_pipeline.py")
    best_model = joblib.load("models/best_model.joblib")
    with open("assets/metrics.json","r") as f:
        metrics = json.load(f)
    return best_model, metrics

best_model, metrics = load_model_and_metrics()

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
    
    st.write("### Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="status", data=df, palette="Set2", ax=ax)
    st.pyplot(fig)
    
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    st.write("### PCA Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", alpha=0.7)
    ax.set_title("PCA Projection")
    st.pyplot(fig)

# --- Tab 2: Models
with tab2:
    st.header("🤖 Model Training & Comparison")
    df_metrics = pd.DataFrame(metrics.items(), columns=["Model","ROC-AUC"]).sort_values(by="ROC-AUC", ascending=False)
    st.dataframe(df_metrics)
    st.bar_chart(df_metrics.set_index("Model"))
    best_name = df_metrics.iloc[0]["Model"]
    st.success(f"🏆 Best Model: {best_name} (ROC-AUC={df_metrics.iloc[0]['ROC-AUC']:.3f})")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix (Best Model)")
    y_pred = safe_predict(best_model, X)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy","Parkinson’s"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve (Best Model)")
    y_pred_prob = safe_predict_proba(best_model, X)[:,1]
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)

# --- Tab 3: Prediction
with tab3:
    st.header("🔮 Prediction")
    mode = st.radio("Choose prediction mode:", ["Safe Mode (slower)", "Fast Mode (faster)"])
    option = st.radio("Choose input type:", ["Manual Input","Upload CSV"])
    
    if option=="Manual Input":
        inputs = {}
        for col in X.columns:
            inputs[col] = st.number_input(col, float(X[col].mean()))
        sample = pd.DataFrame([inputs])
        if st.button("Predict Sample"):
            with st.spinner("Running prediction..."):
                if mode=="Safe Mode (slower)":
                    prob = safe_predict_proba(best_model, sample)[0,1]
                else:
                    prob = fast_predict(best_model, sample)[0]
            st.write("Prediction:", "Parkinson’s" if prob>=0.5 else "Healthy")
            st.progress(float(prob) if mode=="Safe Mode (slower)" else 0.5)
    
    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            new_df = pd.read_csv(file)
            with st.spinner("Running predictions..."):
                if mode=="Safe Mode (slower)":
                    preds = safe_predict(best_model, new_df)
                else:
                    preds = fast_predict(best_model, new_df)
            new_df["Prediction"] = preds
            st.success(f"✅ Predictions completed for {len(new_df)} rows")
            st.dataframe(new_df.head(20))

# --- Tab 4: Train New Model
with tab4:
    st.header("⚡ Train New Model")
    st.write("Upload new dataset to retrain:")
    file = st.file_uploader("Upload CSV for retraining", type=["csv"], key="newtrain")
    if file:
        new_df = pd.read_csv(file)
        st.write("New Data Preview:", new_df.head())
        if st.button("Retrain Models"):
            runpy.run_path("model_pipeline.py")
            st.success("✔️ Models retrained, check if new best model is better!")
            best_model, metrics = load_model_and_metrics()
            st.experimental_rerun()
