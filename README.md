## DSCI_524_team_Cauchy_Kmeans_python 

![](https://github.com/saurav193/DSCI_524_team_Cauchy_Kmeans_python/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/saurav193/DSCI_524_team_Cauchy_Kmeans_python/branch/master/graph/badge.svg)](https://codecov.io/gh/saurav193/DSCI_524_team_Cauchy_Kmeans_python) ![Release](https://github.com/saurav193/DSCI_524_team_Cauchy_Kmeans_python/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/DSCI_524_team_Cauchy_Kmeans_python/badge/?version=latest)](https://DSCI_524_team_Cauchy_Kmeans_python.readthedocs.io/en/latest/?badge=latest)

Saurav, Rob, James, Sree

### Milestone 1 README

For this project, we will be creating python and R packages that implement k-means clustering from scratch. This will work on any dataset with valid numerical features, and includes fit, predict, and score functions, as well as as elbow and silhouette methods for hyperparameter “k” optimization.

+ fit: Outputs a list of cluster centres based on the inputted dataset and k. Only clustered based on valid numerical features.

+ predict: Assigns each point in a dataset to a cluster. Dataset has to be in the same format as the original dataset the model was fit on.

+ score: Outputs a score based on goodness of fit.

+ elbow: Outputs the optimal k hyperparameter using the elbow method

+ silhouette: Outputs the optimal k hyperparameter using the silhouette method.

There is a python package sklearn.cluster.KMeans that has similar functions, and a built in k-means function in R. These packages are not meant to add to the existing ecosystem; they are rather intended to deepen our fundamental understanding of these algorithms.

### Installation:

```
pip install -i https://test.pypi.org/simple/ DSCI_524_team_Cauchy_Kmeans_python
```

### Features
- TODO

### Dependencies

- TODO

### Usage

- TODO

### Documentation
The official documentation is hosted on Read the Docs: <https://DSCI_524_team_Cauchy_Kmeans_python.readthedocs.io/en/latest/>

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
