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
    sample_size = min(300, len(X))
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

# --- Tab 2: Models
with tab2:
    st.header("🤖 Model Training & Comparison")
    df_metrics = pd.DataFrame(metrics.items(), columns=["Model","ROC-AUC"]).sort_values(by="ROC-AUC", ascending=False)
    st.dataframe(df_metrics)
    st.download_button("📥 Download Metrics (CSV)", df_metrics.to_csv(index=False).encode("utf-8"), "metrics.csv", "text/csv")
    st.bar_chart(df_metrics.set_index("Model"))
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

            csv_data = new_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions (CSV)", csv_data, "predictions.csv", "text/csv")

            xlsx_buffer = io.BytesIO()
            with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
                new_df.to_excel(writer, index=False, sheet_name="Predictions")
            st.download_button("📥 Download Predictions (XLSX)", xlsx_buffer.getvalue(), "predictions.xlsx", "application/vnd.ms-excel")

# --- Tab 4: Train New Model
with tab4:
    st.header("⚡ Train New Model")
    file = st.file_uploader("Upload CSV for retraining", type=["csv"], key="newtrain")
    if file:
        new_df = pd.read_csv(file)
        st.write("New Data Preview:", new_df.head())

        if st.button("Retrain Models"):
            new_path = "data/new_train.csv"
            new_df.to_csv(new_path, index=False)
            runpy.run_path("model_pipeline.py")  # pipeline must save best_model_new.joblib + metrics_new.json

            # Load old and new metrics
            old_metrics = metrics
            if os.path.exists("assets/metrics_new.json"):
                with open("assets/metrics_new.json","r") as f:
                    new_metrics = json.load(f)
            else:
                new_metrics = {"NewModel":0.0}

            st.subheader("📊 Comparison of Best Models")
            comp_df = pd.DataFrame({
                "Model": ["Current Best","New Best"],
                "ROC-AUC": [max(old_metrics.values()), max(new_metrics.values())]
            })
            st.dataframe(comp_df)

            st.session_state.new_best_model = "models/best_model_new.joblib"
            st.session_state.new_metrics = new_metrics

            # ROC comparison
            if os.path.exists("models/best_model_new.joblib"):
                new_model = joblib.load("models/best_model_new.joblib")
                y_pred_prob_old = safe_predict_proba(best_model, X)[:,1]
                y_pred_prob_new = safe_predict_proba(new_model, X)[:,1]
                fpr_old, tpr_old, _ = roc_curve(y, y_pred_prob_old)
                fpr_new, tpr_new, _ = roc_curve(y, y_pred_prob_new)
                fig, ax = plt.subplots()
                ax.plot(fpr_old, tpr_old, label="Old Best", color="blue")
                ax.plot(fpr_new, tpr_new, label="New Best", color="green")
                ax.plot([0,1],[0,1],'k--')
                ax.legend()
                st.pyplot(fig)

        if "new_best_model" in st.session_state:
            if st.button("🚀 Promote New Model"):
                shutil.copy(st.session_state.new_best_model, "models/best_model.joblib")
                with open("assets/metrics.json","w") as f:
                    json.dump(st.session_state.new_metrics, f)
                st.success("✅ New model promoted and is now active!")
                st.rerun()
