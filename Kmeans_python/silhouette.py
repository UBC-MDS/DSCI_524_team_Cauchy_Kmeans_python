import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
alt.renderers.enable('notebook')

def silhouette(X, k_array):
    """
    Plots a graph of silhouette scores for each k value
    in the given array using fit. Returns a list of each k value
    in k_array paired with its corresponding silhouette score.

    Parameters
    ----------
    X : 2-d array, shape=(n_samples, n_features)
    The data to be clustered.
    k_array : array
    An array of all contending k values.

    Returns
    -------
    1-d array
    An array containing silhouette scores in the same order as k_array.
    
    Altair chart object
    An Altair chart displaying silhouette scores with their corresponding k values.

    Examples
    --------
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> k_array = [2, 3, 4, 5]
    >>> silhouette(X, k_array)
    
    """

    scores = []
    for i in range(len(k_array)):
        centers, labels = fit(k_array[i])
        score = silhouette_score(X, labels)
        scores.append([k_array[i], score])
        
    scores = pd.DataFrame(scores)
    scores.rename(columns = {0: "k", 1:"Score"}, inplace = True)

    chart = (alt.Chart(scores).mark_line().encode(
        alt.X('k:O', axis = alt.Axis(title = 'k')),
        alt.Y('Score:Q', axis = alt.Axis(title = 'Silhouette score')), 
    ).properties(title = "Silhouette scores", width = 800))
    
    return (scores["Score"], chart)