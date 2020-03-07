import numpy as np
import pandas as pd
from Kmeans_python.predict import predict

def test_predict():
    """Tests for predict"""
    
    #helper data
    X_new = np.random.normal(0, 1, (10, 3))
    centroids = np.array([[1,2,3], [1,6,10], [15,2,6], [1,10,11], [3,10,11]])
    
    # Test cases
    assert predict(X_new, centroids).shape[0] == X_new.shape[0], \
           "Each points in new data needs to be assigned a centroid"
    assert predict(centroids, centroids).shape[0] == np.arange(centroids.shape[0]), \
           "Points with same coordinates as centroids should be assigned to that centroid"
    