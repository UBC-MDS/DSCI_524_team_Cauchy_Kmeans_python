from Kmeans_python.silhouette import silhouette, sil_score
import numpy as np


def test_silhouette():
    """
    Test cases for silhouette
    """
    # Generate data
    X = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
    k_array = [2, 3, 4, 5]
    labels1 = np.array([1, 1, 1, 0, 0, 0])
    labels2 = np.array([1, 1, 1, 1, 1, 0])

    # Generate incorrect data
    X2 = np.array([["sadfsdaf", 1], [2, 2], [3, 3],
                  [10, 10], [11, 11], [12, 12]])
    k_array2 = ["asdf", "asdfs"]

    # Test cases

    # Generate output
    scores, chart = silhouette(X, k_array)

    # Check sil_score with balanced clusters
    assert(sil_score(X, labels1) == 0.850462962962963), \
        "sil_score should return the correct value"

    # Check sil_score edge case with 1 member cluster
    assert(sil_score(X, labels2) == -0.002023809523809539), \
        "sil_score should return the correct value again"

    # Check correct amount of scores are returned
    assert(len(scores) == len(k_array)), "scores are of the wrong length"

    # Error should be thrown for invalid input
    try:
        silhouette(X2, k_array)
        print("Should throw error for non numeric input")
        raise
    except ValueError:
        pass

    # Error should be thrown for invalid input
    try:
        silhouette(X, k_array2)
        print("Should throw error for non numeric input")
        raise
    except ValueError:
        pass
