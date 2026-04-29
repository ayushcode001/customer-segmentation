import pandas as pd
import numpy as np
import logging

def engineer_rfm_features(clean_df):
    """Engineer Recency, Frequency, Monetary and Behavioral features."""
    logging.info("Starting RFM feature engineering...")
    
    # Create the 'is_late_delivery' flag on the fly
    if 'is_late_delivery' not in clean_df.columns:
        clean_df['is_late_delivery'] = (
            clean_df['order_delivered_customer_date'] > clean_df['order_estimated_delivery_date']
        ).astype(int)

    # Define 'today' as the day after the last purchase in the dataset
    max_date = clean_df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    
    # Aggregate using the exact column names generated in Phase 1 & 2
    rfm = (
        clean_df
        .groupby("customer_unique_id")
        .agg(
            Recency=("order_purchase_timestamp", lambda x: (max_date - x.max()).days),
            Frequency=("order_id", "nunique"),
            Monetary=("total_payment_value", "sum"),           # Fixed column name
            AvgReview=("review_score", "mean"),
            AvgInstallments=("payment_installments_max", "mean"), # Fixed column name
            IsLateDelivery=("is_late_delivery", "max")         # Uses the flag we just created
        )
        .reset_index()
    )
    
    # Cast output types to optimize memory footprint
    rfm["Recency"] = rfm["Recency"].astype("Int16")
    rfm["Frequency"] = rfm["Frequency"].astype("Int16")
    rfm["Monetary"] = rfm["Monetary"].astype("float32")
    rfm["AvgReview"] = rfm["AvgReview"].astype("float32")
    rfm["AvgInstallments"] = rfm["AvgInstallments"].astype("float32")
    rfm["IsLateDelivery"] = rfm["IsLateDelivery"].astype("Int8")
    
    logging.info(f"RFM feature engineering complete. Shape: {rfm.shape}")
    return rfm