import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
from config import PROCESSED_DIR, OUTPUTS_DIR, RANDOM_STATE


def train_clustering_pipeline(rfm_df, optimal_k=4):
    """Transformation, Scaling, K-Means Clustering, and PCA."""
    logging.info("Starting clustering pipeline...")
    
    # Isolate numerical features for modeling
    features = ["Recency", "Frequency", "Monetary", "AvgReview", "AvgInstallments"]
    X = rfm_df[features].copy()
    
    # Log-Transform skewed features (Recency, Frequency, Monetary)
    for col in ["Recency", "Frequency", "Monetary"]:
        X[col] = np.log1p(X[col])
        
    # Standardize features to unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit K-Means
    logging.info(f"Fitting K-Means with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate Silhouette Score for validation
    sil_score = silhouette_score(X_scaled, cluster_labels)
    logging.info(f"Model Silhouette Score: {sil_score:.4f}")
    
    # Dimensionality Reduction (PCA) for 2D Visualization in Streamlit/Vue
    logging.info("Applying PCA for visualization projection...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(X_scaled)
    
    # Append results back to the original dataframe
    results_df = rfm_df.copy()
    results_df["Cluster"] = cluster_labels
    results_df["PCA_X"] = pca_coords[:, 0]
    results_df["PCA_Y"] = pca_coords[:, 1]
    
    # Save model artifacts for the FastAPI backend
    models_dir = OUTPUTS_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(kmeans, models_dir / "kmeans_model.pkl")
    joblib.dump(pca, models_dir / "pca_transformer.pkl")
    logging.info("Model artifacts saved.")
    
    return results_df