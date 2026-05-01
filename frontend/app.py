# frontend/app.py
import streamlit as st
import requests
import pandas as pd

import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/predict-segment")
st.set_page_config(page_title="Customer Segmentation UI", layout="wide")

# --- UI Header ---
st.title("Retail Customer Segmentation Dashboard")

# --- Layout: Sidebar for Input ---
st.sidebar.header("Enter Customer Data")

def get_user_input():
    customer_id = st.sidebar.text_input("Customer ID", value="CUST_9999")
    recency = st.sidebar.slider("Recency (Days since last purchase)", 1, 365, 30)
    frequency = st.sidebar.slider("Frequency (Total orders)", 1, 20, 1)
    monetary = st.sidebar.number_input("Monetary Value (Total Spend in BRL)", min_value=1.0, value=150.0)
    review_score = st.sidebar.slider("Average Review Score", 1.0, 5.0, 4.0, step=0.1)
    installments = st.sidebar.slider("Average Installments Used", 1.0, 24.0, 1.0, step=1.0)
    
    return {
        "customer_id": customer_id,
        "recency_days": recency,
        "frequency": frequency,
        "monetary_value": monetary,
        "avg_review_score": review_score,
        "avg_installments": installments
    }

user_data = get_user_input()

# --- Main Action ---
if st.sidebar.button("Predict Customer Segment", type="primary"):
    with st.spinner("Connecting to Model API..."):
        try:
            # Send POST request to FastAPI
            response = requests.post(API_URL, json=user_data)
            
            if response.status_code == 200:
                result = response.json()
                persona = result["persona"]
                
                # Display Results
                st.subheader(f"Classification Result: **{persona}**")
                
                # Dynamic Business Action Logic
                if persona == "Champion":
                    st.success("High Value Customer! Action: Add to VIP list, do NOT offer discounts.")
                elif persona == "Satisfied One-Timer":
                    st.info("Good Potential. Action: Send 10% cross-sell discount to encourage habit.")
                elif persona == "At-Risk Whale":
                    st.warning("Churn Risk! Action: Trigger aggressive 'We Miss You' campaign.")
                elif persona == "Detractor":
                    st.error("Unhappy Customer. Action: Suppress from ads, trigger service recovery email.")
                    
                # Display Raw JSON payload
                with st.expander("View API Response"):
                    st.json(result)
                    
            else:
                st.error(f"API Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Could not connect to the API. Is your FastAPI server running on port 8000?")