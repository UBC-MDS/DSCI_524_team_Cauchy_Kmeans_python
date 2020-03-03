def cluster_summary(X, centroids, cluster_assignments):
    """
    Provides summary of groups created from Kmeans clustering, including centroid coordinates,
    number of data points in training data assigned to each cluster, and within-cluster distance metrics.
    
    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        data on which Kmeans was fit
    centroids : numpy.ndarray
        N-dimensional array containing cluster center locations
    cluster_assignments : array-like
        clusters assigned to each data point in training set

    Returns
    -------
    pandas.DataFrame
        data frame displaying, for each cluster: 
        centroid coordinates, 
        number of data points in training data assigned to each cluster, 
        within-cluster distance metrics
    
    Examples
    --------
    >>> from Kmeans_python import fit, cluster_summary
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, cluster_ass = fit(X, 2)
    >>> cluster_summary(centers, cluster_ass)
    """
    
    df_dict = {}
    for i in range(centroids.shape[1]):
        df_dict[f"x{i+1}"] = centroids[:, i]
    
    num_assigned_list = np.zeros(centroids.shape[1])
    inertia_list = np.zeros(centroids.shape[1])
    for i in np.unique(cluster_ass):
        num_assigned_list[i] = sum(cluster_ass == i)
        inertia_list[i] = np.sum((X[cluster_ass == i] - centroids[i])**2)

    df_dict["Number of assigned training points"] = num_assigned_list
    df_dict["Within cluster inertia"] = inertia_list    
    
    summary_df = pd.DataFrame(data = df_dict)
    summary_df.index.name = "centroid"
     
    return summary_df