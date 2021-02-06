import numpy as np
from math import pi, sqrt, exp, ceil
from scipy.signal import fftconvolve
import h5py

def normalize_data( data ):
    """
    Normalizes the given data with mean = 0 and standard deviation = 1
    
    Arguments:
    -----------
    data         -  1D numpy array
                    raw data
    
    Returns:
    --------
    normalized   -  1D numpy array
                    raw data
             
    """
    assert len(data.shape)==1 or data.shape[1]==1
    data = data.squeeze()
    # Find mean and standard deviation of raw data
    mean = np.mean( data )
    std  = np.std( data )    
    # Normalize the data
    normalized = (data - mean) / std
    
    return normalized


def gaussian(x, mu=0., sigma=1.):
    """
    Returns Gaussian function values for the specified variables x with specified mu and sigma
    
    Arguments:
    -----------
    x         -  int, float, or nD numpy array
                 indepedent variables for the gaussian function
    mu        -  float, default=0.
                 mean of the gaussian distribution
    sigma     -  float, default=1.
                 standard deviation of the gaussian distribution   
                 
    Returns:
    --------
    values of gaussian distribution with mu and sigma for specified values of x
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))


def apply_gaussian_filter( data, size=5, sigma=2. ):
    """
    Applies a Gaussian filter of specified size and specified sigma to the data
    
    Arguments:
    -----------
    data         -  1D numpy array
                    raw data
    size         -  int, default=5
                    size of Gaussian filter, preferably an odd number
    sigma        -  float, default=2.
                    standard deviation of gaussian filter
    Returns:
    --------
    filtered     -  1D numpy array
                    denoised data using gaussian filter
    """
    assert len(data.shape) == 1 or data.shape[1] == 1
    assert size%2 != 0
    data     = data.squeeze()    
    # Create a gaussian filter/kernel with the given arguments
    x        = np.arange( -(size//2), (size//2)+1 )
    kernel   = gaussian( x, mu=0., sigma=sigma )
    # Convolution of the data with gaussian kernel
    filtered = fftconvolve( data, kernel, mode='same' )
    
    return filtered    
    
    
def generate_TS_dataset( data, window_size=20, prediction_horizon=1 ):
    """
    Generates a Time Series (TS) prediction dataset from the given data. The dataset is built with inputs of given window size 
    and targets of given prediction horizon. 
    
    Arguments:
    -----------
    data                  -  1D numpy array
                             one-dimensional preprocessed data 
    window_size           -  int, default=20
                             window size of the input TS
                             must be a multiple of 4
    prediction_horizon    -  int, default=1
                             length of the prediction horizon
                             for one-step TS prediction = 1
                             for multi-step TS prediction = {3,5,7}
    
    Returns:
    --------
    inputs                -  nD numpy array
                             size: n x window_size
                             input TS with row-wise samples
    targets               -  nD numpy array
                             size: n x prediction_horizon
                             target TS with row-wise samples
    """
    n_measurements = len(data)
    inputs  = []
    targets = []
    # Extract inputs and targets from preprocessed data
    for i in range( n_measurements - window_size + 1 - prediction_horizon):
        input_  = list(data[ i : i+window_size ])
        target  = list(data[ i+window_size : i+window_size+prediction_horizon ])
        inputs.append( input_ )
        targets.append( target )
    
    return np.array(inputs), np.array(targets)


def split_data( x, y, split=0.20, shuffle=True ):
    """
    The dataset is shuffled and then split into subsets based on specified fraction with/without shuffling
    
    Arguments:
    ----------
    x                     -  nD numpy array
                             size: n x window_size
                             input TS with row-wise samples
    y                     -  nD numpy array
                             size: n x prediction_horizon
                             target TS with row-wise samples
    split                 -  float, default=0.20                         
                             fraction of the smaller subset (for e.g.: validation/test set)
                          
    Returns:
    --------
    x_train               -  nD numpy array
                             size: n_train x window_size
                             training set - input TS with row-wise samples
    y_train               -  nD numpy array
                             size: n_train x prediction_horizon
                             training set - target TS with row-wise samples
    x_test                -  nD numpy array
                             size: n_test x window_size
                             test set - input TS with row-wise samples
    y_test                -  nD numpy array
                             size: n_test x prediction_horizon
                             test set - target TS with row-wise samples
    """
    n_data  = x.shape[0]
    n_train = ceil( (1-split) * n_data )
    if shuffle is True:
        shuffle = np.random.permutation(n_data)
        x_train = x[ shuffle,:][:n_train,:]
        y_train = y[ shuffle,:][:n_train,:]
        x_test  = x[ shuffle,:][n_train:,:]
        y_test  = y[ shuffle,:][n_train:,:]
        return x_train, y_train, x_test, y_test
    else:
        return x[:n_train,:], y[:n_train,:], x[n_train:,:], y[n_train:,:]
    
def save_hdf5( save_path, x_train, y_train, x_valid, y_valid, x_test, y_test ):
    """
    Saves the data subsets as a hdf5 file to the savepath
    
    Arguments:
    ----------
    save_path             -  str
                             path to save the hdf5 file to
                             e.g.: 'data/temperature-one-step.hdf5'
    x_train               -  nD numpy array
                             size: n_train x window_size
                             training set - input TS with row-wise samples
    y_train               -  nD numpy array
                             size: n_train x prediction_horizon
                             training set - target TS with row-wise samples
    x_valid               -  nD numpy array
                             size: n_valid x window_size
                             test set - input TS with row-wise samples
    y_valid               -  nD numpy array
                             size: n_valid x prediction_horizon
                             test set - target TS with row-wise samples
    x_test                -  nD numpy array
                             size: n_test x window_size
                             test set - input TS with row-wise samples
    y_test                -  nD numpy array
                             size: n_test x prediction_horizon
                             test set - target TS with row-wise samples    
    Returns:
    --------
    
    """
    # Close any open hdf5 file
    try: h5file.close()
    except: pass
    # Create a hdf5 file
    h5file = h5py.File(save_path, 'w')
    # Save the training set
    h5file.create_dataset('train/x', data=x_train, dtype='float64')
    h5file.create_dataset('train/y', data=y_train, dtype='float64')
    # Save the validation set
    h5file.create_dataset('valid/x', data=x_valid, dtype='float64')
    h5file.create_dataset('valid/y', data=y_valid, dtype='float64')
    # Save the test set
    h5file.create_dataset('test/x',  data=x_test, dtype='float64')
    h5file.create_dataset('test/y',  data=y_test, dtype='float64')
    # Close the hdf5 file
    h5file.close()
    print('Datasets have been saved as hdf5 file')