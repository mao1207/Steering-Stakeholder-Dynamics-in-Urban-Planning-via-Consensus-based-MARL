import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

#  Read CSV
df = pd.read_csv('node_pairs_knn4.csv')  

# Create a distance matrix
nodes = np.unique(df[['node', 'node_adj']].values)
n = len(nodes)
large_value = 1e10
dist_matrix = np.full((n, n), large_value)
np.fill_diagonal(dist_matrix, 0)


for idx, row in df.iterrows():
    i, j = np.where(nodes == row['node'])[0], np.where(nodes == row['node_adj'])[0]
    dist_matrix[i, j] = row['distance_m']
    dist_matrix[j, i] = row['distance_m']

# DBSCAN clustering
dbscan = DBSCAN(eps=30, min_samples=2, metric='precomputed')
clusters = dbscan.fit_predict(dist_matrix)

print("Cluster assignments:", clusters)
