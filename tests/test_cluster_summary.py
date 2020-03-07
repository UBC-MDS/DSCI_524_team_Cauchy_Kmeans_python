import numpy as np
import pandas as pd
from Kmeans_python.cluster_summary import cluster_summary

def test_cluster_summary():
    """Tests for cluster_summary"""
    
    # Generate helper data
    X = np.random.normal(0, 1, (10, 3))
    X2 = np.random.normal(0, 1, (10, 4))
    centroids = np.array([[1,2,3], [1,6,10], [15,2,6], [1,10,11], [3,10, 11]])
    centroids2 = np.array([[1,2,3], [1,6,10], [15,2,6], [1,10,11], [3,10,11,12]])
    cluster_assignments = np.array([1, 2, 3, 2, 2, 3, 1, 1, 1, 1])
    cluster_assignments2 = np.array([1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 12])

    # Test cases
    assert cluster_summary(X, centroids, cluster_assignments)["Number of assigned training points"].sum() == cluster_assignments.shape[0], \
            "Total of 'Num assigned training points' should be equal to number of examples in data"
    assert cluster_summary(X, centroids, cluster_assignments).shape[0] == centroids.shape[0], \
            "Number of rows in data frame returned by cluster_summary() should be equal to number of centroids"
    assert (np.array(cluster_summary(X, centroids, cluster_assignments)[["x1", "x2", "x3"]]) == centroids).all(), \
            "First three columns in data frame returned by cluster_summary() should be the centroid coordinates"
    assert (np.array(cluster_summary(X, centroids, cluster_assignments)[["x1", "x2", "x3"]]) == centroids).all(), \
            "First three columns in data frame returned by cluster_summary() should be the centroid coordinates"
    
    try:
        cluster_summary(X2, centroids, cluster_assignments)
        print("Should throw an error if X shape does not match centroid shape")
        raise
    except ValueError:
        pass
    
    try:
        cluster_summary(X, centroids2, cluster_assignments)
        print("Should throw an error for invalid centroid shape")
        raise
    except IndexError:
        pass
    
    try:
        cluster_summary(X, centroids, cluster_assignments2)
        print("Should throw an error if cluster_assignements length does not match number of examples in data")
        raise
    except ValueError:
        pass