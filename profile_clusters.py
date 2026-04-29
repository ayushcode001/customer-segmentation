import pandas as pd
from config import RFM_PATH

def profile_clusters():
    print("\n--- Loading Segmented Data ---")
    df = pd.read_parquet(RFM_PATH)
    
    # Calculate median/mean RFM values per cluster
    profile = df.groupby('Cluster').agg(
        Customer_Count=('customer_unique_id', 'count'),
        Median_Recency_Days=('Recency', 'median'),
        Mean_Frequency=('Frequency', 'mean'),
        Median_Spend_BRL=('Monetary', 'median'),
        Mean_Review=('AvgReview', 'mean')
    ).round(2)
    
    # Calculate percentage of total customer base
    profile['% of Base'] = (profile['Customer_Count'] / len(df) * 100).round(1)
    
    # Reorder columns for readability
    profile = profile[['Customer_Count', '% of Base', 'Median_Recency_Days', 'Mean_Frequency', 'Median_Spend_BRL', 'Mean_Review']]
    
    print("\n--- CLUSTER PROFILES ---")
    print(profile.to_string())

if __name__ == "__main__":
    profile_clusters()