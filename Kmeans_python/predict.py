def predict(X_new, centroids, distance_metric="euclidean"):
    """
    Assigns new data points to clusters based on closest centroid. 
    
    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        New data to assign to clusters
    centroids : numpy.ndarray
        N-dimensional array containing cluster center locations
    distance_metric : string
        distance metric to measure proximity of data to cluster centers.
        Can take on values: "euclidean" (default), "manhattan", "mahalanobis"

    Returns
    -------
    list
        assigned clusters for each point in X_new
    
    Examples
    --------
    >>> from Kmeans_python import fit, predict
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, cluster_ass, distances = fit(X)
    >>> X_test = np.array([[1, 0], [2, 4], [8, 1],
    ...                    [9, 3], [8, 8], [0, 0]])
    >>> predict(X_test, centers)
    """

    pass