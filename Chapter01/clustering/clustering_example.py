import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import datetime
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(123456789)))

# Define simulate ride data function
def simulate_ride_distances():
    """
    Simulates ride distances data.

    Simulates 370 ride distances with normal distribution around 10,
    10 ride distances with normal distribution around 30 (long distances),
    10 ride distances with normal distribution around 10 (same distance),
    and 10 ride distances with normal distribution around 10 (same distance).
    
    Returns:
        ride_dists (numpy.ndarray): An array of simulated ride distances.
    """
    logging.info('Simulating ride distances ...')
    ride_dists = np.concatenate(
        (
            10 * np.random.random(size=370),
            30 * np.random.random(size=10),  # long distances
            10 * np.random.random(size=10),  # same distance
            10 * np.random.random(size=10)  # same distance
        )
    )
    return ride_dists

def simulate_ride_speeds():
    """
    Simulates ride speeds data.

    Simulates 370 ride speeds with normal distribution around 30,
    10 ride speeds with normal distribution around 30 (same speed),
    10 ride speeds with normal distribution around 50 (high speed),
    and 10 ride speeds with normal distribution around 15 (low speed).
    
    Returns:
        ride_speeds (numpy.ndarray): An array of simulated ride speeds.
    """
    logging.info('Simulating ride speeds ...')
    ride_speeds = np.concatenate(
        (
            np.random.normal(loc=30, scale=5, size=370),
            np.random.normal(loc=30, scale=5, size=10), # same speed
            np.random.normal(loc=50, scale=10, size=10), # high speed
            np.random.normal(loc=15, scale=4, size=10) #low speed
        )
    )
    return ride_speeds


def simulate_ride_data():
    """
    Simulates ride data.

    Simulates ride distances, speeds, and times.
    Assembles them into a Data Frame with ride_id as the index.

    Returns:
        df (pandas.DataFrame): A DataFrame containing ride data.
    """
    logging.info('Simulating ride data ...')
    # Simulate some ride data ...
    ride_dists = simulate_ride_distances()
    ride_speeds = simulate_ride_speeds()
    ride_times = ride_dists/ride_speeds

    # Assemble into Data Frame
    df = pd.DataFrame(
        {
            'ride_dist': ride_dists,
            'ride_time': ride_times,
            'ride_speed': ride_speeds
        }
    )
    ride_ids = datetime.datetime.now().strftime("%Y%m%d")+df.index.astype(str)
    df['ride_id'] = ride_ids
    return df



#==========================================
# Clustering with DBSCAN
#==========================================
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics

def plot_cluster_results(data, labels, core_samples_mask, n_clusters_):
    """
    Plots the results of DBSCAN clustering algorithm.

    Parameters
    ----------
    data : numpy.ndarray
        Array of data points to be plotted.
    labels : numpy.ndarray
        Array of labels corresponding to the data points.
    core_samples_mask : numpy.ndarray
        Array of boolean values indicating which data points are core samples.
    n_clusters_ : int
        Estimated number of clusters.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.

    """
    fig = plt.figure(figsize=(10, 10))
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.cool(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], '^', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

    plt.xlabel('Standard Scaled Ride Dist.')
    plt.ylabel('Standard Scaled Ride Time')
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    #plt.show()
    plt.savefig('taxi-rides.png')

def cluster_and_label(data, create_and_show_plot=True):
    """
    Clusters the given data with DBSCAN algorithm and returns the results.

    Parameters
    ----------
    data : numpy.ndarray
        Array of data points to be clustered.
    create_and_show_plot : bool
        Whether to create and show the plot of the clustering results.

    Returns
    -------
    run_metadata : dict
        A dictionary containing the results of the clustering algorithm.
        It includes the estimated number of clusters, the estimated number of noise points,
        the silhouette coefficient, and the labels of the data points.

    """

    data = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=0.3, min_samples=10).fit(data)

    # Find labels from the clustering
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(data, labels))

    run_metadata = {
        'nClusters': n_clusters_,
        'nNoise': n_noise_,
        'silhouetteCoefficient': metrics.silhouette_score(data, labels),
        'labels': labels,
    }
    if create_and_show_plot:
        plot_cluster_results(data, labels, core_samples_mask, n_clusters_)
    else:
        pass
    return run_metadata

if __name__ == "__main__":
    import os
    # If data present, read it in
    file_path = 'taxi-rides.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        logging.info('Simulating ride data')
        df = simulate_ride_data()
        df.to_csv(file_path, index=False)

    # Create some nice plots with create_and_show_plot=True (in a notebook)
    X = df[['ride_dist', 'ride_time']]
    
    logging.info('Clustering and labelling')
    
    results = cluster_and_label(X, create_and_show_plot=True)
    df['label'] = results['labels']
    
    # Output your results to json
    logging.info('Outputting to json ...')
    df.to_json('taxi-labels.json', orient='records')