
import pandas as pd
import numpy as np 
from sklearn import datasets
from sklearn.cluster import KMeans

##Helper data

iris = datasets.load_iris() #loading the iris dataset
features = iris.data #get the input data
labels = iris.target #get the responses, in this case the specie of the flowers


## Test functions