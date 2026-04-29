import pandas as pd
import logging
from config import TARGET_ORDER_STATUS

def clean_data(df):
    '''Data Cleaning '''
    logging.info('Data cleaning')

    initial_rows = len(df)


    # Filter out canceled or unavailable orders
    df = df[df['order_status'].isin(TARGET_ORDER_STATUS)]

    #  Convert datetime columns
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                 'order_estimated_delivery_date', 'shipping_limit_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')


    # Drop rows with missing critical identifiers
    df = df.dropna(subset=['customer_unique_id', 'order_id', 'product_id'])
    
    # Fill missing values for non-critical columns
    df['review_score'] = df['review_score'].fillna(df['review_score'].median())
    df['product_category_name_english'] = df['product_category_name_english'].fillna('unknown')
    
    # Handle anomalies (e.g., zero or negative payments)
    df = df[df['total_payment_value'] > 0]
    df = df[df['price'] > 0]


    # Extract temporal features for EDA
    df['purchase_year'] = df['order_purchase_timestamp'].dt.year
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['purchase_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
    
    final_rows = len(df)
    logging.info(f"Cleaning complete. Dropped {initial_rows - final_rows} rows.")
    return df