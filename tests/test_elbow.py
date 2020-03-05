import numpy as np
import altair as alt
from Kmeans_python.elbow import elbow

def test_elbow():
    """
    Test cases for the elbow function
    """
    # Generate helper data
    X = np.array([[1, 2], [1, 4], [1, 0],
        [10, 2], [10, 4], [10, 0]])
    centers = [2, 3, 4, 5]