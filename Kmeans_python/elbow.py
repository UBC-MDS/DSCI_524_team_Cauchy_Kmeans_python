from Kmeans_python.fit import fit
import numpy as np
import pandas as pd
import altair as alt

def elbow(X, centers_list):
    """
    Creates a plot of inertia vs number of cluster centers
    as per the elbow method. Calculates and returns the inertia 
    values for all cluster centers. Useful for identifying the optimal 
    number of clusters while using k-means clustering algorithm.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
    Input data that is to be clustered.
    centers_list : list or 1-d array-like
    A list of all possible numbers of cluster centers

    Returns
    -------
    tuple
    A tuple of an altair plot object containing a line plot of
    k (number of cluster centers) vs inertia and inertia for all k.

    Examples
    --------
    >>> from Kmeans_python import elbow
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers = [2, 3, 4, 5]
    >>> elbow(X, centers)
    >>> (alt.Chart(...),
        [2.8284271247461903, 2.8284271247461903, 1.4142135623730951, 0.0])    
    """

    