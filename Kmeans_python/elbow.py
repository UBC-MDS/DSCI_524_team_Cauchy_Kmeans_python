from Kmeans_python.fit import fit
import numpy as np
import pandas as pd
import altair as alt

def elbow(X, centers_list):
    """
    Creates a plot of inertia vs number of cluster centers
    as per the elbow method. Calculates and returns the inertia 
    values for all cluster centers. Useful for identifying the optimal 
    number of clusters while using k-means clustering algorithm.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
    Input data that is to be clustered.
    centers_list : list or 1-d array-like
    A list of all possible numbers of cluster centers

    Returns
    -------
    tuple
    A tuple of an altair plot object containing a line plot of
    k (number of cluster centers) vs inertia and inertia for all k.

    Examples
    --------
    >>> from Kmeans_python import elbow
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers = [2, 3, 4, 5]
    >>> elbow(X, centers)
    >>> (alt.Chart(...),
        [2.8284271247461903, 2.8284271247461903, 1.4142135623730951, 0.0])    
    """

    # Ensure input arguments are valid
    if not ((isinstance(X, pd.DataFrame)) | (isinstance(X, np.ndarray))):
        raise ValueError("Invalid input type for samples."
                        " X must be pandas dataframe or a numpy array.")

    if not ((isinstance(centers_list, list)) | (isinstance(centers_list, np.ndarray))):
        raise ValueError("Invalid input type for list of numbers of clusters."
                        " centers_list must be list or a numpy array.")
    
    if (np.min(centers_list) < 1) | (np.max(centers_list) > X.shape[0]):
        raise ValueError("Invalid values in list of numbers of clusters."
                        "Number of clusters should be between 1 and number of samples")

    if not (all(isinstance(k, int) for k in centers_list)):
        raise ValueError("Invalid input type in centers_list",
                         "Number of centers should be of type int")

    # If X is a dataframe, convert it into a numpy array
    if isinstance(X, pd.DataFrame):        
        X = X.to_numpy()
  
    # Iterate through centers list and get inertia
    inertia = []
    for k in centers_list:
        # Fit Kmeans algorithm to get cluster centers and labels
        centers, labels = fit(X, k, n_init = 10, max_iter = 200)
        # Compute inertia
        for cluster in range(k):
            x_cluster = X[np.where(labels == cluster)]
            cluster_inertia = np.linalg.norm(x_cluster - centers[cluster])
        inertia.append(np.sum(cluster_inertia))
    # Save results to a dataframe
    results = pd.DataFrame({"K" : centers_list, "inertia" : inertia})

    # Create a plot object of K vs Inertia 
    p = alt.Chart(results).mark_line().encode(
        alt.X("k:Q", title="k"),
        alt.Y("inertia:Q", title="Inertia")).properties(
        title = "Optimal K Using Elbow Method",
        width = 700,
        height = 300
        ).configure_axis(
        labelFontSize = 20,
        titleFontSize = 20
        ).configure_title(
        fontSize = 20
        )
    return p, inertia