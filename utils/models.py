import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.layers import AveragePooling1D, Conv1D, GRU, Concatenate, Flatten


def simple_LSTM( n_timesteps, n_features, n_output, lstm_units, activations=['selu']):
    """
    Creates a simple LSTM model based on specified arguments.
    Returns an uncompiled model.
    
    Arguments:
    -----------
    n_timesteps     -   int
                        timesteps or window size of the input time series
    n_features      -   int
                        number of features in the input time series
    n_output        -   int
                        timesteps / prediction horizon of the target time series
    lstm_units      -   int
                        number of units in the LSTM layer
    activations     -   list of str
                        activation function for the hidden layers
                        e.g.: 'relu', 'selu', 'tanh', 'sigmoid'
    
    Returns:
    --------
    model           -   uncompiled keras model
                        
    """
    # Layer definitions
    lstm_layer_1 = LSTM(lstm_units[0], activation=activations[0], 
                        return_sequences=True,  name='lstm_1')
    lstm_layer_2 = LSTM(lstm_units[1], activation=activations[1], 
                        return_sequences=True,  name='lstm_2')
    lstm_layer_3 = LSTM(lstm_units[2], activation=activations[2], 
                        return_sequences=False, name='lstm_3')
    output_layer = Dense(n_output, name='output')
    
    # Layer connections
    inputs       = Input(shape=(n_timesteps, n_features), name='input') 
    lstm_1       = lstm_layer_1( inputs )
    lstm_2       = lstm_layer_2( lstm_1 )
    lstm_3       = lstm_layer_3( lstm_2 )
    outputs      = output_layer( lstm_3 )
    
    model = Model(inputs=inputs, outputs=outputs, name='simple_LSTM')
    
    return model


def ACRNN( n_timesteps, n_features, n_output, conv_filters = [32,32,32], conv_kernels = [7,5,3], 
           gru_units = [32,32,32] ):
    """
    Creates the proposed ACRNN model from the paper.
    Returns an uncompiled model.
    
    Arguments:
    -----------
    n_timesteps     -   int
                        timesteps or window size of the input time series
    n_features      -   int
                        number of features in the input time series
    n_output        -   int
                        timesteps / prediction horizon of the target time series
    conv_filters    -   list of ints
                        number of filters for each 1D causal convolutional layer 
    conv_kernels    -   list of ints
                        kernel sizes for each 1D causal convolutional layer 
    gru_units       -   list of ints
                        number of units for each GRU layer
    
    Returns:
    --------
    model           -   uncompiled keras model
    
    """
    assert len(conv_filters) == 3
    assert len(conv_kernels) == 3
    assert len(gru_units   ) == 3
    #=============================================================================================
    ### Layer definitions
    #=============================================================================================
    input_shape     = (n_timesteps, n_features)
    ## Downsampling layers
    downsample_1    = AveragePooling1D( pool_size=2, strides=2, name='downsample_1' )
    downsample_2    = AveragePooling1D( pool_size=4, strides=4, name='downsample_2' )
    ## Causal Convolution Layers
    # Path 1
    convolution_11  = Conv1D( filters=conv_filters[0], kernel_size=conv_kernels[0], padding='causal', 
                             activation='relu', name='convolution_11' )
    convolution_12  = Conv1D( filters=conv_filters[0], kernel_size=conv_kernels[0], padding='causal', 
                             activation='relu', name='convolution_12' )
    # Path 2
    convolution_21  = Conv1D( filters=conv_filters[1], kernel_size=conv_kernels[1], padding='causal', 
                             activation='relu', name='convolution_21' )
    convolution_22  = Conv1D( filters=conv_filters[1], kernel_size=conv_kernels[1], padding='causal', 
                             activation='relu', name='convolution_22' )
    # Path 3
    convolution_31  = Conv1D( filters=conv_filters[2], kernel_size=conv_kernels[2], padding='causal', 
                             activation='relu', name='convolution_31' )
    convolution_32  = Conv1D( filters=conv_filters[2], kernel_size=conv_kernels[2], padding='causal', 
                             activation='relu', name='convolution_32' )
    ## Gated Recurrent Units
    gru_1           = GRU( units=gru_units[0], return_state=True, name='gru_1' )
    gru_2           = GRU( units=gru_units[0], return_state=True, name='gru_2' )
    gru_3           = GRU( units=gru_units[0], return_state=True, name='gru_3' )
    ## Concatenate Layer (along timesteps dimension)
    concat_layer    = Concatenate(axis=1, name='concatenate')
    ## Flatten Layer to shape (None, n_timesteps x n_features)
    flatten         = Flatten()
    ## Linear Layers
    linear_1        = Dense( units=n_output, activation=None, name='linear_1')
    linear_2        = Dense( units=n_output, activation=None, name='linear_2')
    
    #=============================================================================================
    ### Layer Connections
    #=============================================================================================
    inputs          = Input(shape=input_shape, name='input') 
    # Path 1
    conv_11         = convolution_11( inputs )
    conv_12         = convolution_12( conv_11 )
    _, h1           = gru_1( conv_12 )
    # Path 2
    down_1          = downsample_1( inputs )
    conv_21         = convolution_21( down_1 )
    conv_22         = convolution_22( conv_21 )
    _, h2           = gru_2( conv_22 )
    # Path 3
    down_2          = downsample_2( inputs )
    conv_31         = convolution_31( down_2 )
    conv_32         = convolution_32( conv_31 )
    _, h3           = gru_3( conv_32 )
    # Concatenation of final states of GRUs
    concat          = concat_layer([ h1, h2, h3 ])
    # Linear layer 1
    lin_1           = linear_1( concat )
    # Linear layer 2 after flattening 
    flat_inputs     = flatten( inputs )
    lin_2           = linear_2( flat_inputs )
    # Combine outputs 
    outputs         = lin_1 + lin_2
    
    #=============================================================================================
    ### Model
    #=============================================================================================
    model           = Model( inputs=inputs, outputs=outputs, name='ACRNN' )
    
    return model
