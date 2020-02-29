def fit(X_train, k):
    """
    This functions classifies the non-labeled data into a given number of clusters k.
    It returns labels for each data point according to the cluster it belong and also 
    cluster centers. This is a type of unsupervised learning method to classify data.

    Parameters
    ----------
    X_train : numpy.ndarray or a pandas.DataFrame, shape=(n_samples, n_features)
    Input data that is to be clustered with features in the columns and samples in rows

    k : an integer(int)
    The number of clusters we need.

    Returns
    -------
    list
    A list of the centers of each cluster.

    list
    A list of labels for cluster assignment for all samples in the given data

    Examples
    --------
    >>> from Kmeans_python import fit
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, labels = fit(X, 2)
    
    """

    pass