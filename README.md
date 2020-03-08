## Kmeans_python 

![](https://github.com/saurav193/DSCI_524_team_Cauchy_Kmeans_python/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/saurav193/Kmeans_python/branch/master/graph/badge.svg)](https://codecov.io/gh/saurav193/DSCI_524_team_Cauchy_Kmeans_python) ![Release](https://github.com/saurav193/DSCI_524_team_Cauchy_Kmeans_python/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/Kmeans_python/badge/?version=latest)](https://Kmeans_python.readthedocs.io/en/latest/?badge=latest)

Saurav, Rob, James, Sree

### Milestone 2 README

This package will include python packages that implement k-means clustering from scratch. This will work on any dataset with valid numerical features, and includes fit, predict, and cluster_summary functions, as well as elbow and silhouette methods for hyperparameter “k” optimization. A high level overview of each function is given below. See each function's documentation for more details.

+ fit: Takes input data in an nd-array. This function classifies the non-labeled data into a given number of clusters k using simple KMeans algorithm. It returns labels for each data point according to the cluster it belongs and also cluster centers. This is a type of unsupervised learning method to classify data.

+ predict: Takes input data in an ndarray. Assigns each point in a dataset to a cluster. Dataset has to be in the same format as the original dataset the model was fit on.

+ elbow: Creates a plot of inertia vs number of cluster centers as per the elbow method. Calculates and returns the inertia values for all cluster centers. Useful for identifying the optimal number of clusters while using k-means clustering algorithm.

+ silhouette: Returns the average silhouette score of each sample in a given 2-d array and clustering labels.

+ cluster_summary: Provides summary of groups created from Kmeans clustering, including centroid coordinates, number of data points in training data assigned to each cluster, and within-cluster distance metrics.

There is a python package sklearn.cluster.KMeans that has similar functions, and a built in k-means function in R. These packages are not meant to add to the existing ecosystem; they are rather intended to deepen fundamental understanding of these algorithms.

### Installation:

```
pip install -i https://test.pypi.org/simple/Kmeans_python
```

### Tests

To test that the functions work as intended, test files have been written. Use `pytest -q tests/<test_file_name>.py` to test a specific desired function. Automated testing using the `poetry` tool will be introduced in a future release.

### Dependencies

- TODO

### Usage

- TODO

### Documentation
The official documentation is hosted on Read the Docs: <https://Kmeans_python.readthedocs.io/en/latest/>

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
