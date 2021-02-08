import numpy as np
from math import pi, sqrt, exp, ceil
from scipy.signal import fftconvolve
import h5py

def standardize_data( data ):
    """
    Standardizes the given data to have mean = 0 and standard deviation = 1
    
    Arguments:
    -----------
    data         -  1D numpy array
                    raw data
    
    Returns:
    --------
    standardized -  1D numpy array
                    standardized data
    mean         -  float
                    mean of the raw data
    std          -  float
                    standard deviation of the raw data
             
    """
    assert len(data.shape)==1 or data.shape[1]==1
    data = data.squeeze()
    # Find mean and standard deviation of raw data
    mean = np.mean( data )
    std  = np.std( data )    
    # Standardize the data
    standardized = (data - mean) / std
    
    return standardized, mean, std


def normalize_data( data, scale_range=[-1,1]):
    """
    Normalizes the given data between the given range [0,1] or [-1,1]
    
    Arguments:
    -----------
    data         -  1D numpy array
                    raw data
    
    Returns:
    --------
    normalized   -  1D numpy array
                    normalized data
    """
    assert len(data.shape)==1 or data.shape[1]==1
    data = data.squeeze()
    # Find min and max of the data
    min_data = np.min( data )
    max_data = np.max( data )
    # Normalize the data
    normalized = scale_range[0]+((data-min_data)*(scale_range[1]-scale_range[0]))/(max_data-min_data)
    
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


def shuffle_data( x, y ):
    """
    Randomly shuffles the dataset
    
    Arguments:
    ----------
    x                     -  nD numpy array
                             size: n x window_size
                             input TS with row-wise samples
    y                     -  nD numpy array
                             size: n x prediction_horizon
                             target TS with row-wise samples
    
    Returns:
    --------
    x                     -  nD numpy array
                             size: n x window_size
                             input TS with row-wise samples
    y                     -  nD numpy array
                             size: n x prediction_horizon
                             target TS with row-wise samples
    """
    n_data  = x.shape[0]
    shuffle = np.random.permutation(n_data)
    return x[ shuffle,:], y[ shuffle,:] 


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
    
    
def save_hdf5( save_path, x, y, true_mean, true_std ):
    """
    Saves the data subsets as a hdf5 file to the savepath
    
    Arguments:
    ----------
    save_path             -  str
                             path to save the hdf5 file to
                             e.g.: 'data/temperature-one-step.hdf5'
    x                     -  nD numpy array
                             size: n x window_size
                             input TS with row-wise samples
    y                     -  nD numpy array
                             size: n x prediction_horizon
                             target TS with row-wise samples 
    true_mean             -  float
                             mean of the raw dataset before preprocessing
                             helps in projection to true scale
    true_std              -  float
                             std of the raw dataset before preprocessing
                             helps in projection to true scale
    Returns:
    --------
    
    """
    # Close any open hdf5 file
    try: h5file.close()
    except: pass
    # Create a hdf5 file
    h5file = h5py.File(save_path, 'w')
    # Save the preprocessed dataset
    h5file.create_dataset('x', data=x, dtype='float64')
    h5file.create_dataset('y', data=y, dtype='float64')
    # Save the true mean and std of raw dataset
    h5file.create_dataset('true_mean', data=true_mean, dtype='float64')
    h5file.create_dataset('true_std',  data=true_std , dtype='float64')
    # Close the hdf5 file
    h5file.close()
    print('Datasets have been saved as hdf5 file')
    
    
def load_hdf5( file_path ):
    """
    Loads the data subsets from the saved hdf5 file
    
    Arguments:
    ----------
    file_path             -  str
                             path to save the hdf5 file to
                             e.g.: 'data/temperature-one-step.hdf5'
                             
    Returns:
    --------
    x                     -  nD numpy array
                             size: n x window_size
                             input TS with row-wise samples
    y                     -  nD numpy array
                             size: n x prediction_horizon
                             target TS with row-wise samples 
    true_mean             -  float
                             mean of the raw dataset before preprocessing
                             helps in projection to true scale
    true_std              -  float
                             std of the raw dataset before preprocessing
                             helps in projection to true scale
    """
    # Load the hdf5 file
    data           = h5py.File(file_path,'r')
    # Extract the datasets
    x              = data['x'][...]
    y              = data['y'][...]
    true_mean      = data['true_mean'][...]
    true_std       = data['true_std'][...]
    # Close the hdf5 file
    data.close()
    
    return x, y, true_mean, true_std
