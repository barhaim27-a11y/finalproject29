import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import shap

st.set_page_config(page_title="Parkinson’s ML App v28", page_icon="🧠", layout="wide")

# Paths
DATA_PATH = "data/parkinsons.csv"
MODELS_DIR = "models"
ASSETS_DIR = "assets"

# Load dataset
df = pd.read_csv(DATA_PATH)
if "name" in df.columns:
    df = df.drop(columns=["name"])
X = df.drop("status", axis=1)
y = df["status"]

# Load best model + metrics + leaderboard
best_model = joblib.load(os.path.join(MODELS_DIR,"best_model.joblib"))
with open(os.path.join(ASSETS_DIR,"metrics.json")) as f:
    metrics = json.load(f)
with open(os.path.join(ASSETS_DIR,"leaderboard.json")) as f:
    leaderboard = json.load(f)

# =========================
# Helpers
# =========================
def safe_predict_proba(model, X):
    try:
        return model.predict_proba(X)
    except ValueError:
        if hasattr(model,"feature_names_in_"):
            return model.predict_proba(X[model.feature_names_in_])
        raise

def risk_label(prob, threshold=0.5):
    if prob < 0.3: return "🟢 Low"
    elif prob < 0.7: return "🟡 Medium"
    return "🔴 High"

def decision_text(prob, threshold=0.5):
    decision = "Positive (Parkinson’s)" if prob>=threshold else "Negative (Healthy)"
    return f"ההסתברות היא {prob*100:.1f}%, הסיווג עם הסף {threshold:.2f} הוא {decision}"

def plot_shap_values(model, X_sample):
    try:
        explainer = shap.Explainer(model.named_steps["clf"], X_sample)
        shap_values = explainer(X_sample)
        shap.plots.bar(shap_values, max_display=10, show=False)
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data & EDA",
    "🤖 Models",
    "🔮 Prediction",
    "⚡ Train New Model"
])

# --- Tab 1: EDA
with tab1:
    st.header("📊 Data & EDA")
    st.write("Preview:")
    st.dataframe(df.head())
    st.write("Statistical Summary")
    st.dataframe(df.describe().T)

    st.write("Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="status", data=df, palette="Set2", ax=ax)
    st.pyplot(fig)

    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

# --- Tab 2: Models
with tab2:
    st.header("🤖 Model Performance & Zoo")

    # KPIs of best model (test metrics)
    st.subheader("📌 KPIs (Test Set)")
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{metrics['test']['accuracy']:.2f}")
    cols[1].metric("Precision", f"{metrics['test']['precision']:.2f}")
    cols[2].metric("Recall", f"{metrics['test']['recall']:.2f}")
    cols[3].metric("F1", f"{metrics['test']['f1']:.2f}")
    cols[4].metric("ROC-AUC", f"{metrics['test']['roc_auc']:.2f}")

    # Leaderboard
    st.subheader("🏆 Model Zoo")
    leaderboard_df = []
    for name, vals in leaderboard.items():
        leaderboard_df.append({
            "Model": name,
            "ROC-AUC (Test)": vals["test"]["roc_auc"],
            "Accuracy (Test)": vals["test"]["accuracy"]
        })
    leaderboard_df = pd.DataFrame(leaderboard_df).sort_values(by="ROC-AUC (Test)", ascending=False)
    st.dataframe(leaderboard_df)

    choice = st.selectbox("Select a model to visualize:", leaderboard_df["Model"])
    # Simplification: only load best model for now
    model = best_model
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["Healthy","Parkinson’s"]).plot(ax=ax)
    st.pyplot(fig)

    y_prob = safe_predict_proba(model, X)[:,1]
    fpr,tpr,_ = roc_curve(y, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr,tpr,label="ROC")
    ax.plot([0,1],[0,1],'k--')
    st.pyplot(fig)

# --- Tab 3: Prediction
with tab3:
    st.header("🔮 Prediction")
    threshold = st.slider("Decision Threshold",0.0,1.0,0.5,0.01)
    option = st.radio("Input:",["Manual","Upload CSV"])

    if option=="Manual":
        inputs = {col: st.number_input(col, float(X[col].mean())) for col in X.columns}
        sample = pd.DataFrame([inputs])
        if st.button("Predict Sample"):
            prob = safe_predict_proba(best_model, sample)[0,1]
            st.progress(prob)
            st.write(risk_label(prob,threshold))
            st.info(decision_text(prob,threshold))
            if st.button("Explain with SHAP"):
                plot_shap_values(best_model, sample)
    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            new_df = pd.read_csv(file)
            probs = safe_predict_proba(best_model, new_df)[:,1]
            preds = (probs>=threshold).astype(int)
            new_df["Probability"]=probs
            new_df["Prediction"]=preds
            new_df["risk_label"]=[risk_label(p,threshold) for p in probs]
            new_df["decision_text"]=[decision_text(p,threshold) for p in probs]
            st.dataframe(new_df.head())
            csv_data = new_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions",csv_data,"predictions.csv","text/csv")

# --- Tab 4: Train New Model
with tab4:
    st.header("⚡ Train New Model")
    st.write("Upload new dataset to retrain models:")
    file = st.file_uploader("Upload CSV",type=["csv"],key="newtrain")
    if file and st.button("Retrain"):
        new_df = pd.read_csv(file)
        new_df.to_csv("data/new_train.csv",index=False)
        os.system("python model_pipeline.py")  # retrain pipeline
        st.success("Retraining complete! Leaderboard and metrics updated.")

    # Show training log
    log_path = os.path.join(ASSETS_DIR,"training_log.csv")
    if os.path.exists(log_path):
        st.subheader("📜 Training Log")
        log_df = pd.read_csv(log_path)
        st.dataframe(log_df)
        st.download_button("Download Log", log_df.to_csv(index=False).encode("utf-8"), "training_log.csv","text/csv")
