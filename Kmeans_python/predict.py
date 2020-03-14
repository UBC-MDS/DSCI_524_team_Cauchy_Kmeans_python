import numpy as np


def predict(X_new, centroids):
    """
    Assigns new data points to clusters based on closest centroid.

    Parameters
    ----------
    X_new : array-like, shape=(n_samples, n_features)
        New data to assign to clusters
    centroids : numpy.ndarray
        array containing cluster center locations

    Returns
    -------
    numpy.array, shape=(n_samples, )
        assigned clusters for each point in X_new

    Examples
    --------
    >>> from Kmeans_python.fit import fit
    >>> from Kmeans_python.predict import predict
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, cluster_ass = fit(X, 2)
    >>> X_test = np.array([[1, 0], [2, 4], [8, 1],
    ...                    [9, 3], [8, 8], [0, 0]])
    >>> predict(X_test, centers)
    """
    for inputs in [X_new, centroids]:
        if (inputs.dtype != "float" and inputs.dtype != "int"):
            raise TypeError("Input data must be numeric")

    if X_new.shape[1] != centroids.shape[1]:
        raise ValueError("Inputs must have the following shapes: \n \
                          'X':(n, m) \n \
                          'centroids':(k, m)")

    num_examples, num_features = X_new.shape
    dist_sq = (X_new.reshape((num_examples, 1, num_features)) - centroids)**2
    total_distances = np.sum(dist_sq, axis=2)
    return np.argmin(total_distances, axis=1)
