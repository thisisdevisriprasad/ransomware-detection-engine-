import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt

# =========================================
# 🧠 Load Saved Models
# =========================================
@st.cache_resource
def load_models():
    rf = joblib.load("rf_model.joblib")
    kmeans = joblib.load("kmeans_model.joblib")
    pca = joblib.load("pca_model.joblib")
    scaler_sup = joblib.load("scaler_sup.joblib")
    scaler_unsup = joblib.load("scaler_unsup.joblib")
    return rf, kmeans, pca, scaler_sup, scaler_unsup

rf, kmeans, pca, scaler_sup, scaler_unsup = load_models()

# =========================================
# 🧩 Auto Column Alignment
# =========================================
def align_features(df, scaler):
    expected_features = getattr(scaler, "feature_names_in_", None)
    if expected_features is not None:
        missing_cols = [c for c in expected_features if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in expected_features]

        if extra_cols:
            st.warning(f"Dropped {len(extra_cols)} extra columns: {extra_cols}")
            df = df.drop(columns=extra_cols)
        if missing_cols:
            for c in missing_cols:
                df[c] = 0

        df = df.reindex(columns=expected_features, fill_value=0)
    else:
        expected_feature_count = getattr(scaler, "n_features_in_", None)
        if expected_feature_count and df.shape[1] > expected_feature_count:
            df = df.iloc[:, :expected_feature_count]
    return df

# =========================================
# 🧠 Streamlit UI (Simple)
# =========================================
st.set_page_config(page_title="Hybrid Ransomware Detector", layout="centered")

st.title("🧠 AI-Based Hybrid Ransomware Detector")
st.write("Upload a **file’s feature CSV** to automatically detect if it’s ransomware or safe.")
st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload your file (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    X = df.select_dtypes(include=["number"]).fillna(0)
    st.success("✅ File uploaded successfully!")

    if st.button("🚀 Predict"):
        try:
            start = time.perf_counter()

            # Align and scale
            X = align_features(X, scaler_sup)
            X_sup = scaler_sup.transform(X)
            X_unsup = scaler_unsup.transform(X)

            # Supervised (Random Forest)
            sup_probs = rf.predict_proba(X_sup)
            sup_pred = np.argmax(sup_probs, axis=1)

            # Unsupervised (PCA + KMeans)
            X_pca = pca.transform(X_unsup)
            unsup_pred = kmeans.predict(X_pca)

            # Hybrid Logic (65% Supervised + 35% Unsupervised)
            combined = (0.65 * sup_probs[:, 1]) + (0.35 * unsup_pred)
            final_pred = np.where(combined >= 0.5, 1, 0)
            hybrid_score = combined

            # Time and confidence
            latency_ms = (time.perf_counter() - start) / len(X) * 1000
            ransomware_pct = np.mean(hybrid_score) * 100
            safe_pct = 100 - ransomware_pct

            result_labels = ["🟢 Safe" if x == 0 else "🔴 Ransomware" for x in final_pred]
            df["Prediction"] = result_labels

            # =========================================
            # 📊 Display Results
            # =========================================
            st.success(f"✅ Detection Complete! (Latency: {latency_ms:.2f} ms/sample)")
            st.dataframe(df)

            # Confidence Meter
            st.subheader("📈 Confidence Level")
            col1, col2 = st.columns(2)
            col1.metric("🔴 Ransomware Probability", f"{ransomware_pct:.2f}%")
            col2.metric("🟢 Safe Probability", f"{safe_pct:.2f}%")

            # Pie Chart
            counts = pd.Series(result_labels).value_counts()
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax.set_title("Prediction Distribution")
            st.pyplot(fig)

            # Performance Chart
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.bar(["Latency (ms/sample)", "Ransomware %", "Safe %"],
                    [latency_ms, ransomware_pct, safe_pct],
                    color=["skyblue", "red", "green"])
            ax2.set_title("⚡ Real-Time Detection Performance")
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

else:
    st.info("👆 Please upload a CSV file and click **Predict** to begin detection.")

st.caption("Developed by **Pothuganti Devisriprasad** | Hybrid ML (65% Random Forest + 35% KMeans)")
