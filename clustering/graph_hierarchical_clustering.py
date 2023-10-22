import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score as sil_score


def read_csv_fill_dist_matrix(filename):
    df = pd.read_csv(filename)
    nodes = np.unique(df[['node', 'node_adj']].values)
    n = len(nodes)
    large_value = 1e10
    dist_matrix = np.full((n, n), large_value)
    np.fill_diagonal(dist_matrix, 0)

    for idx, row in df.iterrows():
        i, j = np.where(nodes == row['node'])[0][0], np.where(nodes == row['node_adj'])[0][0]
        dist_matrix[i, j] = row['distance_m']
        dist_matrix[j, i] = row['distance_m']

    return dist_matrix, nodes


def determine_best_t(Z, max_clusters, dist_matrix):
    best_t = 2
    best_score = -1
    sil_scores = []

    for num_clusters in range(2, max_clusters+1):
        labels = fcluster(Z, t=num_clusters, criterion='maxclust')
        score = sil_score(dist_matrix, labels)
        sil_scores.append(score)
        if score > best_score:
            best_score = score
            best_t = num_clusters

    return best_t, sil_scores


def plot_elbow(sil_scores, max_clusters):
    plt.plot(range(2, max_clusters+1), sil_scores, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Elbow Method using Silhouette Score')
    plt.savefig('Elbow0.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_negative_landuse_counts(node_info, clusters):
    negative_landuse_counts = {}

    for label, node_ids in clusters.items():
        negative_count = sum(node_info.loc[node_info['node_id'].isin(node_ids), 'landuse_type'] == -1)
        negative_landuse_counts[label] = negative_count

    return negative_landuse_counts

def main():
    # Read CSV and prepare distance matrix
    dist_matrix, nodes = read_csv_fill_dist_matrix('node_pairs_knn4.csv')
    condensed_dist_matrix = squareform(dist_matrix)

    # Hierarchical Clustering
    Z = linkage(condensed_dist_matrix, method='ward', optimal_ordering=True)

    # Elbow method to determine best t
    max_clusters = 10
    best_t, sil_scores = determine_best_t(Z, max_clusters, dist_matrix)
    print(f"The best t value is: {best_t}")

    # Plot the elbow curve
    plot_elbow(sil_scores, max_clusters)

    # Clustering based on best t
    labels = fcluster(Z, best_t, criterion='maxclust')
    clusters = {label: [] for label in set(labels)}
    for node_id, label in enumerate(labels):
        clusters[label].append(node_id)

    # Print the cluster details
    for label, node_ids in clusters.items():
        print(f"Cluster Label: {label}, Node IDs: {node_ids}")

     # Read node_info.csv
    node_info = pd.read_csv('node_info.csv')

    # Get the count of landuse_type = -1 for each cluster label
    negative_landuse_counts = get_negative_landuse_counts(node_info, clusters)
    
    # Print the results
    for label, count in negative_landuse_counts.items():
        print(f"Cluster Label: {label},   Count of nodes with landuse_type = -1: {count}")
    for label, nodes in clusters.items():
        print(f"Label {label} has {len(nodes)} nodes.")
    for label, nodes in clusters.items():
        negative_count = negative_landuse_counts.get(label, 0)  # Get count, default to 0 if not present
        ratio = negative_count / len(nodes)
        
        print(f"Label {label} has a ratio of {ratio:.2f} of nodes with landuse_type = -1.")




    # Plot dendrogram
    plt.figure(figsize=(20, 7))
    dendrogram(Z, labels=nodes)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Node Index')
    plt.ylabel('Distance')
    plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
