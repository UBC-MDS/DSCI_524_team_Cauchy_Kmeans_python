import numpy as np
from Kmeans_python.elbow import elbow
import pandas as pd


def test_elbow():
    """
    Test cases for the elbow function
    """
    # Generate helper data
    X = np.array([[1, 2], [1, 4], [1, 0],
                 [10, 2], [10, 4], [10, 0]])
    centers = [2, 3, 4, 5]

    X_dic = {"x": [1, 2, 3]}

    X_df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1]})

    centers_array = np.array(centers)

    # Test cases
    p, inertia = elbow(X, centers)
    assert(((isinstance(inertia, list)) | (isinstance(inertia, np.ndarray)))),\
        "Inertia is of wrong data type"
    assert(len(inertia) == len(centers)), "Inertia has wrong number of values"
    assert(all(y >= 0 for y in inertia)), \
        "Inertia values should be greater than or equal to zero"
    assert(((isinstance(inertia, list)) | (isinstance(inertia, np.ndarray)))),\
        "Inertia is of wrong data type"

    # check that error is thrown for invalid data shape input
    try:
        elbow(X_dic, centers)
        print("Should throw an error if input data is incompatible")
        raise
    except ValueError:
        pass

    # Should work with dataframe type
    p, inertia = elbow(X_df, centers)
    assert(len(inertia) == len(centers)), "Inertia has wrong number of values"

    # Should work with array type centers
    p, inertia = elbow(X, centers_array)
    assert(len(inertia) == len(centers_array)), "Inertia has wrong number of values"

    # check that error is thrown for invalid number of centers data type
    try:
        elbow(X_dic, ["one", 2])
        print("Should throw an error if input data is incompatible")
        raise
    except ValueError:
        pass

    # check that error is thrown for invalid number of centers data type
    try:
        elbow(X, [2, 3, 100])
        print("Should throw an error if input data is incompatible")
        raise
    except ValueError:
        pass

    # check that error is thrown for invalid number of centers data type
    try:
        elbow(X, np.array([1, 2, 3, 4]))
        print("Should throw an error if input data is incompatible")
        raise
    except ValueError:
        pass

    # check that error is thrown for invalid number of centers data type
    try:
        elbow(X, 5)
        print("Should throw an error if input data is incompatible")
        raise
    except ValueError:
        pass