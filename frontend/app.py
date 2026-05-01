# frontend/app.py
import os
import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# --- Page Config ---
st.set_page_config(page_title="Customer Segmentation UI", layout="wide")

# --- Load Models Directly (no FastAPI needed) ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "outputs" / "models"

@st.cache_resource
def load_models():
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    kmeans = joblib.load(MODELS_DIR / "kmeans_model.pkl")
    pca    = joblib.load(MODELS_DIR / "pca_transformer.pkl")
    return scaler, kmeans, pca

try:
    scaler, kmeans, pca = load_models()
except Exception as e:
    st.error(f"❌ Could not load model files: {e}")
    st.stop()

# Persona Mapping
PERSONA_MAP = {
    0: "Satisfied One-Timer",
    1: "At-Risk Whale",
    2: "Detractor",
    3: "Champion"
}

# --- UI Header ---
st.title("Retail Customer Segmentation Dashboard")

# --- Sidebar Input ---
st.sidebar.header("Enter Customer Data")

def get_user_input():
    customer_id  = st.sidebar.text_input("Customer ID", value="CUST_9999")
    recency      = st.sidebar.slider("Recency (Days since last purchase)", 1, 365, 30)
    frequency    = st.sidebar.slider("Frequency (Total orders)", 1, 20, 1)
    monetary     = st.sidebar.number_input("Monetary Value (Total Spend in BRL)", min_value=1.0, value=150.0)
    review_score = st.sidebar.slider("Average Review Score", 1.0, 5.0, 4.0, step=0.1)
    installments = st.sidebar.slider("Average Installments Used", 1.0, 24.0, 1.0, step=1.0)

    return {
        "customer_id":    customer_id,
        "recency_days":   recency,
        "frequency":      frequency,
        "monetary_value": monetary,
        "avg_review_score": review_score,
        "avg_installments": installments,
    }

user_data = get_user_input()

# --- Predict ---
if st.sidebar.button("Predict Customer Segment", type="primary"):
    with st.spinner("Running prediction..."):
        try:
            # Build feature array (must match training pipeline order)
            features = np.array([[
                np.log1p(user_data["recency_days"]),
                np.log1p(user_data["frequency"]),
                np.log1p(user_data["monetary_value"]),
                user_data["avg_review_score"],
                user_data["avg_installments"],
            ]])

            # Scale → Predict → PCA
            scaled     = scaler.transform(features)
            cluster    = int(kmeans.predict(scaled)[0])
            coords     = pca.transform(scaled)[0]
            persona    = PERSONA_MAP.get(cluster, "Unknown")

            # --- Display Results ---
            st.subheader(f"Classification Result: **{persona}**")

            if persona == "Champion":
                st.success("High Value Customer! Action: Add to VIP list, do NOT offer discounts.")
            elif persona == "Satisfied One-Timer":
                st.info("Good Potential. Action: Send 10% cross-sell discount to encourage habit.")
            elif persona == "At-Risk Whale":
                st.warning("Churn Risk! Action: Trigger aggressive 'We Miss You' campaign.")
            elif persona == "Detractor":
                st.error("Unhappy Customer. Action: Suppress from ads, trigger service recovery email.")

            with st.expander("View Prediction Details"):
                st.json({
                    "customer_id":    user_data["customer_id"],
                    "cluster_id":     cluster,
                    "persona":        persona,
                    "pca_coordinates": {"x": float(coords[0]), "y": float(coords[1])},
                })

        except Exception as e:
            st.error(f"Prediction failed: {e}")