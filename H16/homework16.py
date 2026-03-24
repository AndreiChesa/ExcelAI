from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

df = load_wine(as_frame=True).data

#print(df.head())

kvalues = range(2, 11)
inertia = []

for k in kvalues:
    clustering = KMeans(n_clusters=k, random_state=42)
    clustering.fit(df)
    inertia.append(clustering.inertia_)

silhouette_scores = []

for k in kvalues:
    clustering = KMeans(n_clusters=k, random_state=42)
    cluster_labels = clustering.fit_predict(df)
    silhouette_scores.append(silhouette_score(df, cluster_labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Elbow plot
axes[0].plot(kvalues, inertia, marker='o')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method for Optimal k')
axes[0].grid(True, alpha=0.3)

# Silhouette plot
axes[1].plot(kvalues, silhouette_scores, marker='o', color='orange')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score for Optimal k')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#Normalization

normalized_df = MinMaxScaler().fit_transform(df)

norm_inertia = []

for k in kvalues:
    clustering = KMeans(n_clusters=k, random_state=42)
    clustering.fit(normalized_df)
    norm_inertia.append(clustering.inertia_)

norm_silhouette_scores = []

for k in kvalues:
    clustering = KMeans(n_clusters=k, random_state=42)
    cluster_labels = clustering.fit_predict(normalized_df)
    norm_silhouette_scores.append(silhouette_score(normalized_df, cluster_labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Elbow plot
axes[0].plot(kvalues, norm_inertia, marker='o')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method for Optimal k')
axes[0].grid(True, alpha=0.3)

# Silhouette plot
axes[1].plot(kvalues, norm_silhouette_scores, marker='o', color='orange')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score for Optimal k')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

best_k_raw = kvalues[silhouette_scores.index(max(silhouette_scores))]
best_k_norm = kvalues[norm_silhouette_scores.index(max(norm_silhouette_scores))]
print("Best k (raw):", best_k_raw)
print("Best k (normalized):", best_k_norm)

final_k = best_k_norm
final_clustering = KMeans(n_clusters=final_k, random_state=42)
final_labels = final_clustering.fit_predict(normalized_df)

pca = PCA(n_components=2, random_state=42)
pca_data = pca.fit_transform(normalized_df)

plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=final_labels, cmap='viridis', edgecolor='k', s=100)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'K-Means Clustering (k={final_k}) with PCA')
plt.colorbar(label='Cluster Label')
plt.show()



