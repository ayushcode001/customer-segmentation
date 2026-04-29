import pandas as pd
import logging

def merge_data(dfs):
    ''' Relational join chain avoiding Cartesian explosion'''
    logging.info("Relational merge chain")

    # Order + Customers [Mapping Oder to person]
    base_df = pd.merge(dfs['orders'], dfs['customers'], on='customer_id', how='inner')

    # Aggregate Payments to Order Level before joining
    payment_agg = dfs['payments'].groupby('order_id').agg(
        total_payment_value=('payment_value', 'sum'),
        payment_installments_max = ('payment_installments', 'max'),
        payment_type=('payment_type', lambda x: x.mode()[0] if not x.empty else 'unknown')
    ).reset_index()

    base_df = pd.merge(base_df, payment_agg, on='order_id', how='left')

    # Aggregate Reviews to Order level
    reviews_agg = dfs['reviews'].sort_values('review_creation_date').groupby('order_id').tail(1)
    reviews_agg = reviews_agg[['order_id', 'review_score']]
    base_df = pd.merge(base_df, reviews_agg, on='order_id', how='left')

     
    # Join Items 
    master_df = pd.merge(base_df, dfs['order_items'], on='order_id', how='left')  

    # Join Products
    master_df = pd.merge(master_df, dfs['products'], on='product_id', how='left')

    # Join Product Category Translation
    master_df = pd.merge(master_df, dfs['translation'], on='product_category_name', how='left')


    # Join sellers 
    master_df = pd.merge(master_df, dfs['sellers'], on='seller_id', how='left')
    
    logging.info(f"Merge complete. Master shape: {master_df.shape}")
    return master_df

