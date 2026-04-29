import logging
import pandas as pd
from src.ingestion import load_data
from src.merge import merge_data
from src.cleaning import clean_data
from src.eda import perform_eda
from src.features import engineer_rfm_features
from src.model import train_clustering_pipeline
from config import MASTER_TXN_PATH, CLEAN_TXN_PATH, RFM_PATH

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- Starting Full Olist ML Pipeline ---")
    
    try:
        # Data Engineering
        dfs = load_data()
        master_df = merge_data(dfs)
        clean_df = clean_data(master_df)
        
        # Save checkpoints
        master_df.to_parquet(MASTER_TXN_PATH, index=False)
        clean_df.to_parquet(CLEAN_TXN_PATH, index=False)
        
        # EDA
        perform_eda(clean_df)
        
        # Feature Engineering
        rfm_df = engineer_rfm_features(clean_df)
        
        # Modeling
        segmented_customers = train_clustering_pipeline(rfm_df, optimal_k=4)
        
        # Save final segmented dataset
        segmented_customers.to_parquet(RFM_PATH, index=False)
        
        logging.info("--- Pipeline Execution Successful ---")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()