import pandas as pd
import numpy as np


def compute_distance(samples, centers):

    """
    This computes the euclidean distance of each sample from all the
    cluster centers

    Parameters
    ----------

    samples : numpy.ndarray
    all the data points in the sample

    centers : numpy.ndarray
    the centroids of the clusters already selected

    Returns
    ---------

    numpy.ndarray
    an array with all the samples and their distances from each of the
    cluster centers
    """

    k, num_features = centers.shape
    n_samples = samples.shape[0]

    distance = np.empty((n_samples, k))

    for i in range(k):

        # taking the euclidean distance
        distance[:, i] = np.linalg.norm(samples - centers[i, :], axis=1)

    return distance


def fit(X_train, k, n_init=10, max_iter=200):

    """
    This function classifies the non-labeled data into a given number of
    clusters k using simple KMeans algorithm.
    It returns labels for each data point according to the cluster
    it belongs and also
    cluster centers. This is a type of unsupervised learning
    method to classify data.

    Parameters
    ----------
    X_train : numpy.ndarray or a pandas.DataFrame,
    shape=(n_samples, n_features)
    Input data that is to be clustered with features in the columns
    and samples in rows

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

    test_df = isinstance(X_train, pd.DataFrame)

    if not (test_df | (isinstance(X_train, np.ndarray))):
        raise ValueError("Invalid input type for samples. \
            X_train must be pandas dataframe or a numpy array.")

    if (n_init <= 0) | (not (isinstance(n_init, int))):
        raise ValueError(f"Invalid number of initializations \
            : {n_init} must be bigger than zero.")

    if (max_iter <= 0) | (not (isinstance(max_iter, int))):
        raise ValueError("Number of interations should be a positive integer."
                         " max_iter=%d must be bigger than zero." % max_iter)

    if (k < 1) | (not (isinstance(k, int))):
        raise ValueError("Invalid number of clusters."
                         " k=%d must be an integer >= 1." % k)

    if isinstance(X_train, pd.DataFrame):

        X = X_train.to_numpy()
    else:
        X = X_train
    n_samples, n_features = X.shape

    labels = np.empty((n_samples, 1))

    inertia = np.inf

    i = 0

    while i <= n_init:

        # randomly initializing centers
        centers = X[np.random.choice(range(n_samples), k, replace=False), :]

        for j in range(max_iter):

            # calculating euclidean distance of each sample from the centers
            centroid_distances = compute_distance(X, centers)
            labels = np.argmin(centroid_distances, axis=1)

            # updating the centers with mean
            for m in range(k):

                centers[m, :] = np.mean(X[np.where(labels == m)[0], :], axis=0)

        sq_min = np.square(np.min(centroid_distances, axis=1))
        updated_inertia = np.sum(sq_min, axis=0)

        # keeping the best possible values of clusters
        if updated_inertia < inertia:

            inertia = updated_inertia
            centers_final = centers
            labels_final = labels

            if inertia == 0:
                break

        i = i + 1

    return centers_final, labels_final
