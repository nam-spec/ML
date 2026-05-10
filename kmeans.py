# ==========================================
# K-MEANS CLUSTERING
# MALL CUSTOMER DATASET
# Google Colab Ready Code
# ==========================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# Step 2: Upload Dataset
# ==========================================

from google.colab import files
uploaded = files.upload()

# ==========================================
# Step 3: Load Dataset
# ==========================================

df = pd.read_csv('Mall_Customers.csv')

print(df.head())


# ==========================================
# Step 4: Select Features
# ==========================================

# Using Annual Income and Spending Score

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ==========================================
# Step 5: Feature Scaling
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# Step 6: Elbow Method
# ==========================================

wcss = []

for i in range(1, 11):
    
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )
    
    kmeans.fit(X_scaled)
    
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph

plt.figure(figsize=(8,5))

plt.plot(range(1,11), wcss, marker='o')

plt.title('Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('WCSS')

plt.show()

# ==========================================
# Step 7: Train KMeans Model
# ==========================================

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

# Predict clusters
y_kmeans = kmeans.fit_predict(X_scaled)

# ==========================================
# Step 8: Visualize Clusters
# ==========================================

plt.figure(figsize=(10,6))

plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=y_kmeans
)

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    marker='X'
)

plt.title('Customer Segments')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.show()

# ==========================================
# Step 9: Add Cluster Labels to Dataset
# ==========================================

df['Cluster'] = y_kmeans

print(df.head())