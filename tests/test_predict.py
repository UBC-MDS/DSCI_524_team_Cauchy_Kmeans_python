import numpy as np
from Kmeans_python.predict import predict


def test_predict():
    """Tests for predict"""

    # helper data
    X_new = np.random.normal(0, 1, (10, 3))
    centroids = np.array([[1, 2, 3], [1, 6, 10], [15, 2, 6], [1, 10, 11],
                          [3, 10, 11]])
    centroids2 = np.array([["A", 2, 3], [1, 6, 10], [15, 2, 6], [1, 10, 11],
                          [3, 10, 11]])

    # test to check that all new points are assigned a cluster
    assert predict(X_new, centroids).shape[0] == X_new.shape[0], \
        "Each points in new data needs to be assigned a centroid"

    # test to check that points with same coords as cluster are assigned to
    # that cluster
    predicted_clusters = predict(centroids, centroids)
    assert (predicted_clusters == np.arange(centroids.shape[0])).all(), \
        """Points with same coordinates as centroids
        should be assigned to that centroid"""

    # check that error is thrown for invalid input datatype
    try:
        predict(X_new, centroids2)
        print("Should throw an error for invalid" +
              " input data")
        raise
    except TypeError:
        pass
