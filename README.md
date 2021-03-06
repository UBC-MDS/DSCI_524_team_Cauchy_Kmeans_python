# Kmeans_python 

![](https://github.com/UBC-MDS/Kmeans_python/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/Kmeans_python/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/Kmeans_python) ![Release](https://github.com/UBC-MDS/Kmeans_python/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/kmeans-python/badge/?version=latest)](https://kmeans-python.readthedocs.io/en/latest/?badge=latest)

## The Team 

In no specific order:

> - Saurav Chowdhury
>
> - Robert Blumberg
>
> - James Huang
>
> - Sreejith Munthikodu


### Package description

This package includes python functions that implement k-means clustering from scratch. This will work on any dataset with valid numerical features, and includes fit, predict, and cluster_summary functions, as well as elbow and silhouette methods for hyperparameter “k” optimization. A high level overview of each function is given below. See each function's documentation for more details.

+ fit: Takes input data in an nd-array. This function classifies the non-labeled data into a given number of clusters k using simple KMeans algorithm. It returns labels for each data point according to the cluster it belongs and also cluster centers. This is a type of unsupervised learning method to classify data.

+ predict: Takes input data in an ndarray. Assigns each point in a dataset to a cluster. Dataset has to be in the same format as the original dataset the model was fit on.

+ elbow: Creates a plot of inertia vs number of cluster centers as per the elbow method. Calculates and returns the inertia values for all cluster centers. Useful for identifying the optimal number of clusters while using k-means clustering algorithm.

+ silhouette: Returns the average silhouette score of each sample in a given 2-d array and clustering labels.

+ cluster_summary: Provides summary of groups created from Kmeans clustering, including centroid coordinates, number of data points in training data assigned to each cluster, and within-cluster distance metrics.

There is a python package sklearn.cluster.KMeans that has similar functions, and a built in k-means function in R. These packages are not meant to add to the existing ecosystem; they are rather intended to deepen fundamental understanding of the Kmeans algorithms.

### Dependencies
- pandas == 1.0.1
- numpy == 1.18.1
- altair == 4.0.1
- scikit-learn == 0.22.1

### Installation:

The package has been deployed to test pypi. If you do not have the dependencies listed above installed, please use the command below.
```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple kmeans-python
```

You may also install the package using the command below if the dependecicies are already installed.   
```
pip install -i https://test.pypi.org/simple/ kmeans-python
```

### Tests

To test that the functions work as intended, test files have been written. You need to run `pip install -U pytest`.  

Use `poetry run pytest` to test all the functions, or `pytest -q tests/<test_file_name>` to test a specific function

### Usage
Simple examples for running each function are shown below.

- **fit.py**
```python    
    from Kmeans_python.fit import fit    
    import numpy as np    
    import pandas as pd    
    X = np.array([[1, 2], [1, 4], [1, 0],    
                  [10, 2], [10, 4], [10, 0]])    
    centers, labels = fit(X, 2)  
```
- **predict.py**
```python    
    from Kmeans_python.fit import fit
    from Kmeans_python.predict import predict
    import numpy as np    
    X = np.array([[1, 2], [1, 4], [1, 0],    
                   [10, 2], [10, 4], [10, 0]])    
    centers, cluster_ass = fit(X, 2)  
    X_test = np.array([[1, 0], [2, 4], [8, 1],  
                        [9, 3], [8, 8], [0, 0]])  
    predict(X_test, centers)  
    >>> array([1, 1, 0, 0, 0, 1])
```
- **elbow.py**
```python  
   from Kmeans_python.elbow import elbow  
   import numpy as np  
   X = np.array([[1, 2], [1, 4], [1, 0],  
                 [10, 2], [10, 4], [10, 0]])  
   centers = [2, 3, 4, 5]
   elbow(X, centers)  
   >>> (alt.Chart(...), [2.8284271247461903, 0.0, 0.0, 0.0])
```
- **silhouette.py**
```python  
   from Kmeans_python.fit import fit
   from Kmeans_python.silhouette import silhouette  
   import numpy as np
   X = np.array([[1, 2], [1, 4], [1, 0],  
                   [10, 2], [10, 4], [10, 0]])  
   k_array = [2, 3, 4, 5]  
   silhouette(X, k_array)  
   >>> (0    0.713348
        1    0.436301
        2    0.166667
        3    0.083333
      Name: Score, dtype: float64,
      alt.Chart(...))
```
- **cluster_summary.py**
```python  
    from Kmeans_python.fit import fit
    from Kmeans_python.cluster_summary import cluster_summary  
    import numpy as np  
    import pandas as pd  
    X = np.array([[1, 2], [1, 4], [1, 0],  
                  [10, 2], [10, 4], [10, 0]])  
    centers, cluster_ass = fit(X, 2)  
    cluster_summary(X, centers, cluster_ass)  
    >>> 
                  x1  x2  Number of assigned training points  Within cluster inertia
 centroid                                                                    
        0          1   2                                 3.0                     8.0
        1         10   2                                 3.0                     8.0
```

### Documentation
The official documentation is hosted on Read the Docs: https://kmeans-python.readthedocs.io/en/latest/

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
