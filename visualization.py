import numpy as np
from sklearn.cluster import KMeans
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import pickle

with open('compress_arr.pkl', 'rb') as file:
    doc = pickle.load(file)


n_components = 2  # Reduce to 2 dimensions for visualization
svd = TruncatedSVD(n_components=n_components)
kmeans = KMeans(n_clusters=7)  # Define the number of clusters

# Create a pipeline: TruncatedSVD followed by KMeans
pipeline = make_pipeline(svd, kmeans)

# Fit the pipeline to the data
pipeline.fit(doc)

# Get the cluster labels
cluster_labels = pipeline.predict(doc)

# Visualize clusters in a 2D space
plt.figure(figsize=(8, 6))
plt.scatter(svd.fit_transform(doc)[:, 0], svd.fit_transform(doc)[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Clustering of Data with TruncatedSVD and KMeans')
plt.xlabel('SVD Component 1')
plt.ylabel('SVD Component 2')
plt.colorbar(label='Cluster')
plt.show()
