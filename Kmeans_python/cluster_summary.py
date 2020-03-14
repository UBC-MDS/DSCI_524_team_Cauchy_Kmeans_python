import numpy as np
import pandas as pd


def cluster_summary(X, centroids, cluster_assignments):
    """
    Provides summary of groups created from Kmeans clustering,
    including centroid coordinates, number of data points in training
    data assigned to each cluster, and within-cluster distance metrics.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        data on which Kmeans was fit
    centroids : numpy.ndarray
        N-dimensional array containing cluster center locations
    cluster_assignments : array-like
        clusters assigned to each data point in training set

    Returns
    -------
    pandas.DataFrame
        data frame displaying, for each cluster:
        centroid coordinates,
        number of data points in training data assigned to each cluster,
        within-cluster distance metrics

    Examples
    --------
    >>> from Kmeans_python.fit import fit
    >>> from Kmeans_python.cluster_summary import cluster_summary
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, cluster_ass = fit(X, 2)
    >>> cluster_summary(centers, cluster_ass)
    """
    for inputs in [X, centroids, cluster_assignments]:
        if (inputs.dtype != "float" and inputs.dtype != "int"):
            raise TypeError("Input data must be numeric")

    if np.max(cluster_assignments) > centroids.shape[0]:
        raise ValueError("Cannot have a cluster assignment \
                         greater than the total number of clusters")

    try:
        df_dict = {}
        for i in range(centroids.shape[1]):
            df_dict[f"x{i+1}"] = centroids[:, i]
        num_centroids = centroids.shape[0]
        num_assigned_list = np.zeros(num_centroids)
        inertia_list = np.zeros(num_centroids)
        for i in range(num_centroids):
            num_assigned_list[i] = sum(cluster_assignments == i)
            squared_dist = (X[cluster_assignments == i] - centroids[i, :])**2
            inertia_list[i] = np.sum(squared_dist)

    except IndexError:
        print("Inputs must have the following shapes:\n" +
              "'X':(n, m), 'centroids':(k, m), 'cluster_assignments':(n, )")
        raise

    df_dict["Number of assigned training points"] = num_assigned_list
    df_dict["Within cluster inertia"] = inertia_list

    summary_df = pd.DataFrame(data=df_dict)
    summary_df.index.name = "centroid"

    return summary_df
