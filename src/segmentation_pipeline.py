"""
Customer Segmentation Pipeline (RFM + K-Means + PCA)
Author: Swagat Gharat
Portfolio: https://swagat-portfolio.onrender.com
Tech Stack: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def load_or_generate_data(file_path: str = None) -> pd.DataFrame:
    """Loads transaction data from CSV or generates a realistic synthetic dataset if not found."""
    if file_path is None or not os.path.exists(file_path):
        candidates = [
            file_path,
            "data/online_retail_transactions.csv",
            "../data/online_retail_transactions.csv",
            "data/online_retail.csv",
            "../data/online_retail.csv",
            "data/data.csv",
            "../data/data.csv",
            os.path.join(os.path.dirname(__file__), "..", "data", "online_retail_transactions.csv"),
            os.path.join(os.path.dirname(__file__), "..", "data", "online_retail.csv"),
            os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")
        ]
        for c in candidates:
            if c and os.path.exists(c):
                file_path = c
                break

    if file_path and os.path.exists(file_path):
        print(f"[1/6] Loading transactional dataset from {file_path}...")
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        return df

    print("[1/6] Transaction file not found. Generating synthetic retail transaction dataset...")
    np.random.seed(42)
    n_records = 15000
    n_customers = 1200

    customer_ids = np.random.choice(range(10001, 10001 + n_customers), size=n_records)
    base_date = pd.to_datetime("2024-01-01")
    random_days = np.random.randint(0, 365, size=n_records)
    invoice_dates = [base_date + pd.Timedelta(days=int(d)) for d in random_days]
    quantities = np.random.choice([1, 2, 3, 5, 10, 20], size=n_records, p=[0.4, 0.25, 0.15, 0.1, 0.07, 0.03])
    unit_prices = np.round(np.random.exponential(scale=15.0, size=n_records) + 2.0, 2)

    df = pd.DataFrame({
        "InvoiceNo": [f"INV{100000 + i}" for i in range(n_records)],
        "CustomerID": customer_ids,
        "InvoiceDate": invoice_dates,
        "Quantity": quantities,
        "UnitPrice": unit_prices
    })
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans transaction data by removing returns, null customers, and non-positive values."""
    print("[2/6] Cleaning raw transaction records...")
    df = df.copy()

    # Drop null customer IDs
    initial_rows = len(df)
    df = df.dropna(subset=["CustomerID"])

    # Cast CustomerID to integer then string
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

    # Filter out cancelled orders (typically starts with 'C')
    if df["InvoiceNo"].dtype == object:
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Positive quantities and prices only
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Calculate Total Spend
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    print(f"      Rows cleaned: {initial_rows:,} -> {len(df):,} transactions across {df['CustomerID'].nunique():,} unique customers.")
    return df


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Recency, Frequency, and Monetary (RFM) metrics per customer."""
    print("[3/6] Engineering Recency, Frequency & Monetary (RFM) features...")
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda d: (snapshot_date - d.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum")
    ).reset_index()

    # Filter any non-positive monetary values
    rfm = rfm[rfm["Monetary"] > 0].reset_index(drop=True)
    return rfm


def plot_rfm_distributions(rfm: pd.DataFrame, output_dir: str = "output"):
    """Plots and saves raw vs log-transformed RFM distributions to inspect skewness correction."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    for idx, col in enumerate(["Recency", "Frequency", "Monetary"]):
        # Raw distribution
        sns.histplot(rfm[col], kde=True, ax=axes[0, idx], color="#1f77b4")
        axes[0, idx].set_title(f"Raw {col} Distribution", fontsize=11, fontweight="bold")
        axes[0, idx].grid(True, linestyle="--", alpha=0.4)

        # Log-transformed distribution
        log_vals = np.log1p(rfm[col])
        sns.histplot(log_vals, kde=True, ax=axes[1, idx], color="#2ca02c")
        axes[1, idx].set_title(f"Log-Transformed log({col} + 1)", fontsize=11, fontweight="bold")
        axes[1, idx].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Raw vs. Log-Transformed RFM Feature Distributions (Skewness Correction)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "rfm_distributions_skewness.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Saved distribution skewness plot to: {plot_path}")


def find_optimal_k(X_scaled: np.ndarray, max_k: int = 8, output_dir: str = "output"):
    """Plots the Elbow Method (Inertia) and Silhouette scores for k=2..max_k."""
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(k_range, inertias, marker="o", color="#1f77b4", linewidth=2, label="Inertia (Elbow / WCSS)")
    ax1.set_xlabel("Number of Clusters (k)", fontweight="bold")
    ax1.set_ylabel("Within-Cluster Sum of Squares (Inertia)", color="#1f77b4", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(k_range, silhouette_scores, marker="s", color="#ff7f0e", linewidth=2, linestyle="--", label="Silhouette Score")
    ax2.set_ylabel("Mean Silhouette Coefficient", color="#ff7f0e", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    plt.title("Optimal Cluster Evaluation: Elbow Method & Silhouette Coefficient", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "optimal_k_selection.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Saved optimal k evaluation plot to: {plot_path}")


def plot_snake_plot(rfm_scaled: np.ndarray, rfm: pd.DataFrame, output_dir: str = "output"):
    """Generates a standardized snake plot showing relative RFM centroids across personas."""
    rfm_norm = pd.DataFrame(rfm_scaled, columns=["Recency", "Frequency", "Monetary"])
    rfm_norm["Persona"] = rfm["Persona"]
    melted_rfm = pd.melt(rfm_norm, id_vars=["Persona"], var_name="Metric", value_name="Standardized_Value")

    plt.figure(figsize=(10, 5))
    palette = {
        "Champions (Super VIPs)": "#107c41",
        "Loyal Steady Buyers": "#0078d4",
        "Recent / Potential Loyalists": "#f59e0b",
        "Dormant / At-Risk": "#d83b01"
    }
    sns.lineplot(
        data=melted_rfm,
        x="Metric",
        y="Standardized_Value",
        hue="Persona",
        palette=palette,
        marker="o",
        linewidth=2.5
    )
    plt.title("Customer Segment RFM Profiles (Snake Plot)", fontsize=13, fontweight="bold")
    plt.ylabel("Standardized Score (Z-Score)", fontweight="bold")
    plt.xlabel("RFM Feature Dimension", fontweight="bold")
    plt.axhline(0, color="gray", linestyle="--", alpha=0.6)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "snake_plot_clusters.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Saved snake plot to: {plot_path}")


def plot_pareto_revenue(profile_table: pd.DataFrame, output_dir: str = "output"):
    """Generates dual-bar Pareto chart comparing Customer Share % against Revenue Contribution %."""
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.35
    x = np.arange(len(profile_table))

    b1 = ax.bar(x - bar_width/2, profile_table["Customer_Share_%"], bar_width, label="Customer Share (%)", color="#94a3b8")
    b2 = ax.bar(x + bar_width/2, profile_table["Revenue_Share_%"], bar_width, label="Revenue Share (%)", color="#107c41")

    ax.set_xticks(x)
    ax.set_xticklabels(profile_table["Persona"], rotation=15, ha="right", fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Customer Base vs. Revenue Contribution (The Pareto Principle)", fontsize=13, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    for bar in b1:
        ax.annotate(f"{bar.get_height():.1f}%", (bar.get_x() + bar.get_width()/2, bar.get_height() + 1), ha='center', fontsize=9)
    for bar in b2:
        ax.annotate(f"{bar.get_height():.1f}%", (bar.get_x() + bar.get_width()/2, bar.get_height() + 1), ha='center', fontsize=9, fontweight='bold', color="#107c41")

    plt.ylim(0, 80)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cluster_revenue_breakdown.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Saved Pareto revenue breakdown chart to: {plot_path}")


def run_pipeline(k_clusters: int = 4, data_path: str = None):
    """Executes the complete end-to-end customer segmentation workflow."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("=================================================================")
    print("        CUSTOMER SEGMENTATION & RFM CLUSTERING PIPELINE          ")
    print("=================================================================")

    # 1. Load & clean
    raw_df = load_or_generate_data(data_path)
    clean_df = clean_data(raw_df)

    # 2. Compute RFM
    rfm = calculate_rfm(clean_df)

    # 3. Distribution Visualization & Log Transformation
    plot_rfm_distributions(rfm, output_dir=output_dir)
    rfm_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])

    # 4. Standard Scaling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # 5. Evaluate K (Elbow & Silhouette)
    print("[4/6] Evaluating cluster models across k=2..8...")
    find_optimal_k(rfm_scaled, max_k=8, output_dir=output_dir)

    # 6. Fit K-Means
    print(f"[5/6] Fitting K-Means++ model with optimal k={k_clusters}...")
    kmeans = KMeans(n_clusters=k_clusters, init="k-means++", random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # Dynamic Persona Assignment
    cluster_stats = rfm.groupby("Cluster").agg(
        Recency_Mean=("Recency", "mean"),
        Frequency_Mean=("Frequency", "mean"),
        Monetary_Mean=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum"),
        Customer_Count=("CustomerID", "count")
    )

    vip_id = cluster_stats["Monetary_Mean"].idxmax()
    dormant_id = cluster_stats["Recency_Mean"].idxmax()
    remaining = [c for c in cluster_stats.index if c not in (vip_id, dormant_id)]
    new_buyer_id = min(remaining, key=lambda c: cluster_stats.loc[c, "Recency_Mean"])
    loyal_id = [c for c in remaining if c != new_buyer_id][0]

    persona_map = {
        vip_id: "Champions (Super VIPs)",
        loyal_id: "Loyal Steady Buyers",
        new_buyer_id: "Recent / Potential Loyalists",
        dormant_id: "Dormant / At-Risk"
    }
    rfm["Persona"] = rfm["Cluster"].map(persona_map)

    # 7. Summary Profile Table
    profile_table = rfm.groupby("Persona").agg(
        Customer_Count=("CustomerID", "count"),
        Avg_Recency_Days=("Recency", "mean"),
        Avg_Frequency_Orders=("Frequency", "mean"),
        Avg_Monetary_Spend=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum")
    ).reset_index()

    profile_table["Customer_Share_%"] = (profile_table["Customer_Count"] / len(rfm) * 100).round(1)
    profile_table["Revenue_Share_%"] = (profile_table["Total_Revenue"] / rfm["Monetary"].sum() * 100).round(1)
    profile_table = profile_table.sort_values(by="Total_Revenue", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 78)
    print("                  CUSTOMER SEGMENT BEHAVIORAL PROFILES                 ")
    print("=" * 78)
    print(profile_table.to_string(index=False))
    print("=" * 78)

    # 8. Visualizations: Snake Plot & Pareto Chart
    plot_snake_plot(rfm_scaled, rfm, output_dir=output_dir)
    plot_pareto_revenue(profile_table, output_dir=output_dir)

    # 9. 2D PCA Dimensionality Reduction
    print("[6/6] Applying PCA (2D Projection) & saving segment exports...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(rfm_scaled)
    rfm["PCA1"] = pca_coords[:, 0]
    rfm["PCA2"] = pca_coords[:, 1]

    plt.figure(figsize=(9.5, 6))
    palette = {
        "Champions (Super VIPs)": "#107c41",
        "Loyal Steady Buyers": "#0078d4",
        "Recent / Potential Loyalists": "#f59e0b",
        "Dormant / At-Risk": "#d83b01"
    }
    sns.scatterplot(
        data=rfm,
        x="PCA1",
        y="PCA2",
        hue="Persona",
        palette=palette,
        alpha=0.7,
        s=45
    )
    plt.title(f"Customer Clusters Visualized via 2D PCA (k={k_clusters})", fontsize=13, fontweight="bold")
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance Explained)", fontweight="bold")
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance Explained)", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    pca_plot_path = os.path.join(output_dir, "pca_clusters_2d.png")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Saved 2D PCA cluster projection plot to: {pca_plot_path}")

    # 10. Save Segmented CSV
    output_csv = os.path.join(output_dir, "segmented_customers.csv")
    rfm.to_csv(output_csv, index=False)
    print(f"      Saved segmented customer dataset to: {output_csv}")
    print("\n[SUCCESS] Pipeline executed cleanly. All visual assets and datasets generated.")


if __name__ == "__main__":
    run_pipeline(k_clusters=4)
