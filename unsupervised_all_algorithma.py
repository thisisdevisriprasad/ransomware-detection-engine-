import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, MeanShift,
    OPTICS, Birch, SpectralClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

# ============================================================
# 🗂️ Setup Output Directory
# ============================================================
output_dir = "eda_unsupervised_output"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 🧾 Load Dataset
# ============================================================
file_path = "data_file_unlabeled.csv"
df = pd.read_csv(file_path)
print("✅ Dataset Loaded — Shape:", df.shape)

# ============================================================
# 🕵️ EDA (Exploratory Data Analysis)
# ============================================================

# Basic info
with open(os.path.join(output_dir, "dataset_info.txt"), "w") as f:
    f.write("Shape: " + str(df.shape) + "\n\n")
    f.write(str(df.info()) + "\n\n")
    f.write("Summary Statistics:\n")
    f.write(str(df.describe()) + "\n\n")
    f.write("Missing Values:\n")
    f.write(str(df.isnull().sum()) + "\n")

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

# Pairplot of first few features
sns.pairplot(df.iloc[:, :5])
plt.suptitle("Pairplot of First 5 Features", y=1.02)
plt.savefig(os.path.join(output_dir, "pairplot.png"))
plt.close()

# ============================================================
# ⚙️ Data Preprocessing
# ============================================================

df_numeric = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ============================================================
# ⚡ Helper Function for Evaluation
# ============================================================

def evaluate_clusters(labels, data):
    if len(set(labels)) > 1:
        sil = silhouette_score(data, labels)
        ch = calinski_harabasz_score(data, labels)
        db = davies_bouldin_score(data, labels)
    else:
        sil, ch, db = np.nan, np.nan, np.nan
    return sil, ch, db

results = []

# ============================================================
# 🤖 Apply All Unsupervised Algorithms
# ============================================================

algorithms = {
    "KMeans": KMeans(n_clusters=3, random_state=42),
    "DBSCAN": DBSCAN(eps=1.5, min_samples=5),
    "Agglomerative": AgglomerativeClustering(n_clusters=3),
    "GaussianMixture": GaussianMixture(n_components=3, random_state=42),
    "MeanShift": MeanShift(),
    "OPTICS": OPTICS(min_samples=5),
    "Birch": Birch(n_clusters=3),
    "SpectralClustering": SpectralClustering(n_clusters=3, random_state=42, assign_labels='kmeans')
}

for name, model in algorithms.items():
    print(f"🔹 Running {name}...")
    try:
        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X_scaled)
        else:
            labels = model.fit(X_scaled).predict(X_scaled)

        sil, ch, db = evaluate_clusters(labels, X_scaled)
        results.append([name, sil, ch, db])

        # Save PCA 2D Cluster Plot
        plt.figure(figsize=(6,4))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=35)
        plt.title(f"{name} Clustering (PCA 2D)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_clusters.png"))
        plt.close()

    except Exception as e:
        print(f"⚠️ {name} failed: {e}")
        results.append([name, np.nan, np.nan, np.nan])

# ============================================================
# 📈 Save and Display Results
# ============================================================

results_df = pd.DataFrame(results, columns=["Algorithm", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"])
results_path = os.path.join(output_dir, "unsupervised_results.csv")
results_df.to_csv(results_path, index=False)

print("\n===== 🧾 Clustering Results =====")
print(results_df)
print(f"\nAll outputs saved in: {os.path.abspath(output_dir)}")
