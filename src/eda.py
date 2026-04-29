import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from config import PLOTS_DIR, REPORTS_DIR


def perform_eda(df):
    """visual insights and text reports."""
    logging.info("Generating EDA plots and reports...")
    sns.set_theme(style="whitegrid")
    
    with open(REPORTS_DIR / "eda_summary.txt", "w") as f:
        f.write("=== E-Commerce Dataset EDA Summary ===\n\n")
        f.write(f"Total Transactions: {len(df)}\n")
        f.write(f"Unique Customers: {df['customer_unique_id'].nunique()}\n")
        f.write(f"Total Revenue: R$ {df['price'].sum():,.2f}\n")
        f.write(f"Date Range: {df['order_purchase_timestamp'].min()} to {df['order_purchase_timestamp'].max()}\n\n")



    # Orders over time
    plt.figure(figsize=(14, 6))
    df.set_index('order_purchase_timestamp').resample('ME')['order_id'].nunique().plot()
    plt.title('Monthly Order Volume')
    plt.ylabel('Unique Orders')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_monthly_orders.png")
    plt.close()


    # Top 10 Product Categories by Revenue
    plt.figure(figsize=(10, 6))
    top_cats = df.groupby('product_category_name_english')['price'].sum().nlargest(10)
    sns.barplot(y=top_cats.index, x=top_cats.values, hue=top_cats.index, palette='viridis', legend=False)
    plt.title('Top 10 Product Categories by Revenue')
    plt.xlabel('Total Revenue (R$)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_top_categories_revenue.png")
    plt.close()


    # Payment Type Distribution
    plt.figure(figsize=(8, 8))
    payment_counts = df['payment_type'].value_counts()
    plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Payment Type Distribution')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_payment_distribution.png")
    plt.close()


    # Review Score Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='review_score', hue='review_score', palette='coolwarm', legend=False)
    plt.title('Distribution of Review Scores')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_review_scores.png")
    plt.close()

    logging.info(f"EDA complete. Plots saved to {PLOTS_DIR}")