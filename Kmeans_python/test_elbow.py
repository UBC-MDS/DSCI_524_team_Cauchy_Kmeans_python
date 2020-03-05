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

    # Test cases
    p, inertia = elbow(X, centers)
    assert(((isinstance(inertia, list)) | (isinstance(inertia, np.ndarray)))), "Inertia is of wrong data type"
    assert(len(inertia) == len(centers)), "Inertia has wrong number of values"
    assert(isinstance(p, alt.vegalite.v3.api.Chart)), "Altair plot object is not returned"
    assert(all(y>=0 for y in inertia)), "Inertia values should be greater than or equal to zero"
