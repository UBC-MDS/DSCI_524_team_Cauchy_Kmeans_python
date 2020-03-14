import numpy as np
from Kmeans_python.cluster_summary import cluster_summary


def test_cluster_summary():
    """Tests for cluster_summary"""

    # Generate helper data
    X = np.random.normal(0, 1, (10, 3))
    X2 = np.random.normal(0, 1, (10, 4))
    centroids = np.array([[1, 2, 3], [1, 6, 10], [15, 2, 6], [1, 10, 11],
                          [3, 10, 11]])
    centroids2 = np.array([[1, 2, 3, 13], [1, 6, 10, 13], [15, 2, 6, 13],
                           [1, 10, 11, 14], [3, 10, 11, 12]])
    centroids3 = np.array([["A", 2, 3], [1, 6, 10], [15, 2, 6], [1, 10, 11],
                          [3, 10, 11]])
    cluster_assignments = np.array([1, 2, 3, 2, 2, 3, 1, 1, 1, 1])
    cluster_assignments2 = np.array([1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 1])

    # test to check that number of assigned points is the same as the number
    # of given points
    cluster_summ = cluster_summary(X, centroids, cluster_assignments)
    n_assigned = cluster_summ["Number of assigned training points"].sum()
    assert n_assigned == cluster_assignments.shape[0], \
        """Total of 'Num assigned training points'
        must equal number examples in data"""

    # test to check that a row is given in dataframe returned
    # by cluster_summary for each centroid
    cluster_summ = cluster_summary(X, centroids, cluster_assignments)
    assert cluster_summ.shape[0] == centroids.shape[0], \
        """Number of rows in data frame returned by cluster_summary()
        should be equal to number of centroids"""

    # check that the first three columns in dataframe are the centroid coords
    cluster_summ = cluster_summary(X, centroids, cluster_assignments)
    assert (np.array(cluster_summ[["x1", "x2", "x3"]]) == centroids).all(), \
        """First three columns in data frame returned by cluster_summary() \n\
        should be the centroid coordinates"""

    # check that error is thrown for invalid data shape input
    try:
        cluster_summary(X2, centroids, cluster_assignments)
    except ValueError:
        pass

    # check that error is thrown for invalid centroid shape input
    try:
        cluster_summary(X, centroids2, cluster_assignments)
    except ValueError:
        pass

    # check that error is thrown for invalid cluster_assignments shape input
    try:
        cluster_summary(X, centroids, cluster_assignments2)
    except IndexError:
        pass

    # check that error is thrown for invalid input datatype
    try:
        cluster_summary(X, centroids3, cluster_assignments)
    except TypeError:
        pass
