
import pandas as pd
import numpy as np 
from sklearn import datasets
from Kmeans_python.fit import fit

## Test function for center
def test_edge():

    test_df = pd.DataFrame({'X1' : np.zeros(10), 'X2' : np.ones(10)} )

    centers, labels = fit(test_df, 1)

    print(labels)
    
    assert centers.all() == np.array([0,1]).all()

## Test function for center
def test_center():

    ##Helper data

    iris = datasets.load_iris() #loading the iris dataset
    features = iris.data 
    labels = iris.target 
    test_df1 = pd.DataFrame({'X1' : features[:,2], 'X2' : features[:,3]} )

    test_df2 = pd.DataFrame({'X1' : np.arange(9), 'X2': np.arange(9)})
    
    ## getting centers of clusters
    centers1, _ = fit(test_df2, 1)

    centers2, _ = fit(test_df1, 2)

    assert centers2.all() == np.array([[4.92525253, 1.68181818], [1.49215686, 0.2627451]]).all(), "Centers did not match"

    assert centers1.all() == np.array([4,4]).all(), "Centers did not match"

## Test function for labels
def test_label():

    ##Helper data
    test_df3 = pd.DataFrame({'X1' : np.concatenate((np.arange(5,10), np.arange(15,20)), axis = 0) , 'X2':  np.concatenate((np.arange(5,10), np.arange(15,20)), axis = 0)})
    
    ## getting the labels for the helper data
    _ , labels = fit(test_df3, 2)

    assert labels.all() == np.concatenate((np.zeros(5), np.ones(5)), axis = 0).all(), "labels did not match"

def test_exceptions():

    ##Helper data
    test_df4 = "this is a python package"
    test_df2 = pd.DataFrame({'X1' : np.arange(9), 'X2': np.arange(9)})
    K = -2
    num_init = 0
    max_iteration = 4.5

    ## checking the exception handling of the function
    try:
        fit(test_df4, 2)
        print("Should throw an error if data is not in a dataframe or numpy array")
        raise
    except ValueError:
        pass
    
    try:
        fit(test_df2, K)
        print("Should throw an error for invalid number of clusters")
        raise
    except ValueError:
        pass
    
    try:
        fit(test_df2, 1, n_init = num_init)
        print("Should throw an error if number of initializations is set to 0")
        raise
    except ValueError:
        pass

    try:
        fit(test_df2, 1, max_iter= max_iteration)
        print("Should throw an error if number of iterations is not integer")
        raise
    except ValueError:
        pass