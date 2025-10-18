from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np


def cluster_and_label(X):
    """
    Clusters the given data with DBSCAN algorithm and returns the results.

    Parameters
    ----------
    X : numpy.ndarray
        Array of data points to be clustered.

    Returns
    -------
    run_metadata : dict
        A dictionary containing the results of the clustering algorithm.
        It includes the estimated number of clusters, the estimated number of noise points,
        the silhouette coefficient, and the labels of the data points.
    """
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)

    # Find labels from the clustering
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    run_metadata = {
        'nClusters': n_clusters_,
        'nNoise': n_noise_,
        'silhouetteCoefficient': metrics.silhouette_score(X, labels),
        'labels': labels,
    }
    return run_metadata
