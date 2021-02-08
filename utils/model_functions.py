import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import KFold
from dtaidistance import dtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Setting Matplotlib default parameters
plt.rcParams.update( {'font.size':14} )

def kfold_cv( k, x, y, true_mean, true_std, model_func, model_args, compile_kwargs, train_kwargs ):
    """
    Performs k-fold cross validation by training the given model on k-folds of the dataset with (k-1)
    folds used for training and 1 fold used for testing. The training step is written as another helper
    function 'train_model' below. The metrics on the test set are reported for the k different models.
    
    Arguments:
    ----------
    k                     -   int
                              number of folds to divide the dataset into.
    x                     -   nD numpy array
                              size: n x window_size
                              input TS with row-wise samples
    y                     -   nD numpy array
                              size: n x prediction_horizon
                              target TS with row-wise samples 
    true_mean             -   float
                              mean of the raw dataset before preprocessing
                              helps in projection to true scale
    true_std              -   float
                              std of the raw dataset before preprocessing
                              helps in projection to true scale
    model_func            -   function handle
                              function to load the model
                              found in utils/models.py
    model_args            -   list
                              arguments needed to create the model with model_func
                              
    compile_kwargs:
    loss                  -   str or keras loss object
                              loss function used for training
                              e.g.: 'mse', or tf.keras.losses.MSE
    metrics               -   str (or) list of str (or) keras metrics object
                              evaluation metrics
                              e.g. 'mae' or tf.keras.metrics.MeanAbsoluteError
    optimizer             -   str or keras optimizer object
                              optimizer for training
                              e.g. 'adam' or tf.keras.optimizers.Adam
                              
    train_kwargs:
    epochs                -   int
                              number of training epochs
    batch_size            -   int, default=32
                              size of each batch during training
    stopping_patience     -   int, default=50
                              number of epochs without improvement before which training is stopped
    dataset               -   str
                              name of the dataset
                              e.g.: 'temperature' or 'sunspot'
                              
    Returns:
    --------
    evaluation_metrics    -   dict
                              loss and metrics evaluated for the k different models
    
    """
    # Initialize lists to store evaluation metrics
    mse_folds = []
    mae_folds = []
    dtw_folds = []
    # Initialize the k-fold cross validator
    kfold = KFold(n_splits=k, shuffle=True)
    # Loop over k-folds to train and evaluate the model
    fold = 1
    for train, test in kfold.split(x, y):
        # Create and compile the model
        model = model_func( *model_args )
        model.compile( **compile_kwargs )
        if fold == 1:
            model.summary()  
            print(110*'=')
        print('Starting k-fold cross validation for fold {}'.format(fold) )
        # Train the model
        model = train_model( model, x[train], y[train], fold=fold, **train_kwargs)        
        ## Evaluate model on test set
        x_test = x[test]
        y_test = y[test]
        scores = model.evaluate( x[test], y[test], verbose=0 )
        mse_folds.append( scores[0] )
        mae_folds.append( scores[1] )
        # Print metrics
        print('MSE on test set for fold {} = {:.6f}'.format(fold, scores[0]) )
        print('MAE on test set for fold {} = {:.6f}'.format(fold, scores[1]) )
        # Compute Dynamic Time Warping in the case of multi-step TS prediction
        if y.shape[1] > 1:
            y_pred = model.predict( x_test )
            dtw_score = compute_dtw( y_test, y_pred )
            dtw_folds.append( dtw_score )
            print('DTW on test set for fold {} = {:.4f}'.format(fold, dtw_score ) )     
        # Move onto the next fold
        fold += 1
        print(110*'=')
    print('Final results after k-fold cross validation:')
    print('MSE = {:.6f} +/- {:.6f}'.format( np.mean(mse_folds), np.std(mse_folds) ) )
    print('MAE = {:.6f} +/- {:.6f}'.format( np.mean(mae_folds), np.std(mae_folds) ) )
    if y.shape[1] > 1:
        print('DTW = {:.4f} +/- {:.4f}'.format( np.mean(dtw_folds), np.std(dtw_folds) ) )
        
        return {'mse': mse_folds, 'mae': mae_folds, 'dtw': dtw_folds }
    else:
        return {'mse': mse_folds, 'mae': mae_folds }
    
    
def compute_dtw( true, pred ):
    """
    Computes the Dynamic Time Warping using the FastDTW algorithm.
    References:
    [1]. Stan Salvador, and Philip Chan. “FastDTW: Toward accurate dynamic time warping in 
         linear time and space.” Intelligent Data Analysis 11.5 (2007): 561-580.
    [2]. https://pypi.org/project/fastdtw/
    
    Arguments:
    ----------
    true                  -   nD numpy array
                              size: n x prediction_horizon
                              target TS with row-wise samples 
    pred                  -   nD numpy array
                              size: n x prediction_horizon
                              predicted TS with row-wise samples 
    
    Returns:
    --------
    dtw_score             -   float
                              distance measure computed from DTW
    """
    # 1. FastDTW
    dtw_score    = 0
    for sample in range( true.shape[0] ):
        dtw_sample, _ = fastdtw( true[sample,:], pred[sample,:], dist=euclidean )
        dtw_score += dtw_sample
    dtw_score    = dtw_score / true.shape[0]
    
    ## 2. Independent DTW for multi-dimensional TS of shape = (n_samples, n_timesteps)
    ## Ref: https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html#multi-dimensionsal-dtw
    #dtw_score = 0
    #for sample in range( pred.shape[0]):
    #    dtw_score += dtw.distance( true[sample,:], pred[sample,:] )
       
    return dtw_score
    
    
def train_model(model, x_train, y_train, fold=None, epochs=100, batch_size=32, stopping_patience=50, dataset=None):
    """
    Trains the model based on the given configuration using the training and validation sets.
    Early Stopping is used to monitor improvement of validation loss with specified stopping patience.
    A checkpoint callback is used to keep track of and save the weights of the model at the
    epoch with minimum validation loss. The best model is saved as a keras h5 object.
    The evolution of losses during training is visualized.
    
    Returns the model after being restored with the weights from the best epoch.
    
    Arguments:
    ----------
    model               -   keras model object
                            compiled model to be trained
    x_train             -   nD numpy array
                            size: n_train x window_size
                            training set - input TS with row-wise samples
    y_train             -   nD numpy array
                            size: n_train x prediction_horizon
                            training set - target TS with row-wise samples
    fold                -   int
                            fold number during k-fold cross validation
                        
    epochs              -   int
                            number of training epochs
    batch_size          -   int, default=32
                            size of each batch during training
    stopping_patience   -   int, default=50
                            number of epochs without improvement before which training is stopped
    dataset             -   str
                            name of the dataset
                            e.g.: 'temperature' or 'sunspot'
     
    Returns:
    --------
    model           -   keras model object
                        trained model with the weights from best epoch reloaded
    """
    # Configuration for saving the weights of the best model
    save_path = 'trained_models/' + dataset + '_' + model.name + '_' + str(y_train.shape[1]) + '-step/' 
    try:
        os.mkdir(save_path)
    except:
        pass
    save_path           += 'fold_' + str(fold) + '_'
    checkpoint_path     = save_path + 'weights.hdf5'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(   filepath=checkpoint_path,
                                                                save_weights_only=True,
                                                                monitor='val_loss',
                                                                mode='min',
                                                                save_best_only=True )
    # Early Stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=stopping_patience)

    # Training the LSTM model
    print('Training the {} model for {} epochs'.format(model.name, epochs))
    start_time  = time.time()
    train_log   = model.fit( x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                             validation_split=0.10, callbacks=[checkpoint_callback, early_stopping])
    end_time    = time.time()
    print('Time taken for training: {:.3f} s'.format(end_time - start_time) )
    # Load the weights of the best epoch
    model.load_weights( checkpoint_path )
    # Save the best model itself
    model_path  = save_path + 'model.h5'
    model.save(model_path)

    # Saving the training log as csv file
    log_path    = save_path + 'training_log.csv'
    pd.DataFrame.from_dict(train_log.history).to_csv(log_path,index=False)
    
    # Quantities to plot
    train_loss = train_log.history['loss']
    valid_loss = train_log.history['val_loss']
    best_epoch = valid_loss.index( min(valid_loss) )
    print('Saved weights and model from epoch {}'.format(best_epoch))

    # Plotting
    fig,ax = plt.subplots(figsize=(15,10))
    plt.grid(True)
    plt.semilogy( train_loss, 'b', label='training loss' )
    plt.semilogy( valid_loss, 'r', label='validation loss' )
    plt.axvline( x=best_epoch, color='k', label='best epoch: {}'.format(best_epoch+1), lw=1.25, ls=(0,(5,5)))
    plt.axhline( y=valid_loss[best_epoch], color='r', label='min. validation loss', lw=2, ls=(0,(1,3)))
    plt.annotate( s=r'min. valid. loss = {:.4f}'.format(valid_loss[best_epoch]), 
                  xy=(0,0.95*valid_loss[best_epoch]), c='r', ha='left',va='top')
    plt.xlabel('Epochs [-]')
    plt.ylabel('Mean Squared Error [-]')
    ax.tick_params(which='minor', labelsize=9)
    ax.tick_params(which='major')
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    plt.legend(loc='upper right')
    plt.title('Evolution of losses during training of {} model'.format(model.name) )
    fig_path = save_path + 'losses.pdf'
    plt.savefig(fig_path)
    plt.show()
    
    return model
