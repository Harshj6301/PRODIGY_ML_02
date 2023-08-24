import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Set title and description
st.title("KMeans Clustering Web App")
st.write("This app performs KMeans clustering and visualizes the clusters.")

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Number of clusters
num_clusters = st.slider("Select the number of clusters:", 2, 10, 4)

# Fit KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# Get cluster centers
centers = kmeans.cluster_centers_

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(num_clusters):
    ax.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i+1}')
ax.scatter(centers[:, 0], centers[:, 1], marker='x', color='black', s=100, label='Cluster Centers')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Display the plot
st.pyplot(fig)

# Display cluster assignments
st.write("Cluster Assignments:")
st.write(labels)
