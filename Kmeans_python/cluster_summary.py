def cluster_summary(centroids, cluster_assignments):
    """
    Provides summary of groups created from Kmeans clustering, including centroid coordinates,
    number of data points in training data assigned to each cluster, and within-cluster distance metrics.
    
    Parameters
    ----------
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
    >>> from Kmeans_python import fit, cluster_summary
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, cluster_ass = fit(X, 2)
    >>> cluster_summary(centers, cluster_ass,)
    """

    pass