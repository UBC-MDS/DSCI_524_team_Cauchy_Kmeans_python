from Kmeans_python.silhouette import silhouette, sil_score
import numpy as np

def test_silhouette():
    """
    Test cases for silhouette
    """
    # Generate data
    X = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
    k_array = [2, 3, 4, 5]
    labels1 = np.array([1,1,1,0,0,0])
    labels2 = np.array([1,1,1,1,1,0])

    # Test cases
    scores, chart = silhouette(X, k_array)
    assert(sil_score(X, labels1) == 0.850462962962963), "sil_score should return the correct value"
    assert(sil_score(X, labels2) == -0.002023809523809539), "sil_score should return the correct value again"
    assert(len(scores) == len(k_array)), "scores are of the wrong length"
    #assert(isinstance(chart, alt.vegalite.v3.api.Chart)), "Should have returned an altair plot"