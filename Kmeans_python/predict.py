import numpy as np

def predict(X_new, centroids, distance_metric="euclidean"):
    """
    Assigns new data points to clusters based on closest centroid. 
    
    Parameters
    ----------
    X_new : array-like, shape=(n_samples, n_features)
        New data to assign to clusters
    centroids : numpy.ndarray
        array containing cluster center locations
    distance_metric : string
        distance metric to measure proximity of data to cluster centers.
        Can take on values: "euclidean" (default), "manhattan", "mahalanobis"

    Returns
    -------
    numpy.array, shape=(n_samples, )
        assigned clusters for each point in X_new
    
    Examples
    --------
    >>> from Kmeans_python import fit, predict
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, cluster_ass = fit(X, 2)
    >>> X_test = np.array([[1, 0], [2, 4], [8, 1],
    ...                    [9, 3], [8, 8], [0, 0]])
    >>> predict(X_test, centers)
    """
    
    num_examples, num_features = X_new.shape
    return np.argmin(np.sum((X_new.reshape((num_examples, 1 , num_features)) - centroids)**2, axis=2), axis=1)