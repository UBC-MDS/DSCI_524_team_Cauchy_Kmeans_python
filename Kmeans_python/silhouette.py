def silhouette(X, k_array):
    """
    Plots a graph of silhouette scores for each k value
    in the given array using fit. Returns a list of each k value
    in k_array paired with its corresponding silhouette score.

    Parameters
    ----------
    X : 2-d array, shape=(n_samples, n_features)
    The data to be clustered.
    k_array : array
    An array of all contending k values.

    Returns
    -------
    2-d array
    An array containing each k value paired with a score.

    Examples
    --------
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> k_array = [2, 3, 4, 5]
    >>> silhouette(X, k_array)
    
    """

    pass
