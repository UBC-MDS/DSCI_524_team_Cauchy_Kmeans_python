def elbow(X, centers):
    """
    Creates a plot of inertia vs number of cluster centers
    as per the elbow method. Returns the inertia values for 
    all cluster centers. Useful for identifying the optimal 
    number of clusters while using k-means clustering algorithm.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
    Input data that is to be clustered.
    centers : list
    A list of all possible numbers of cluster centers

    Returns
    -------
    list
    A list of inertia values for all numbers of cluster
    centers

    Examples
    --------
    >>> from Kmeans_python import elbow
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers = [2, 3, 4, 5]
    >>> elbow(X, centers)
    
    """

    pass
