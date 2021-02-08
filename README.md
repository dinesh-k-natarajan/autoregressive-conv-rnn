# autoregressive-conv-rnn

Implementation of the 2019 [paper](https://arxiv.org/abs/1903.02540) 'Autoregressive Convolutional Recurrent Neural Network for Univariate and Multivariate Time Series Prediction' by Matteo Maggiolo and Gerasimos Spanakis.


## 1. Datasets 
Univariate datasets for time-series (TS) forecasting:  
  * Daily minimum temperatures in Melbourne [(Source)](https://www.kaggle.com/paulbrabban/daily-minimum-temperatures-in-melbourne)  
  * Monthly sunspot number in Zurich [(Source)](https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Data%20Files/monthly-sunspot-number-zurich-17.csv)  
  
### 1.1. Data preprocessing
The authors use the following preprocessing steps:  
  * Per-variable normalization with mu=0 and sigma=1  
  * Denoising using gaussian filter of size=5 and sigma=2  
  
Denoising after normalization lead to higher standard deviation in the data such as sigma=3.8. Therefore, in order to preserve sigma=1 after denoising, the order of the preprocessing steps have been reversed in this implementation. It also lead to drastically improved MSE to the order of 10, i.e., ACRNN improved from MSE=0.2119 to MSE=0.01564 with the reversed preprocessing steps.

Implementation can be found in: [`preprocessing_temperature.ipynb`](https://github.com/dinesh-k-natarajan/autoregressive-conv-rnn/blob/main/preprocessing_temperature.ipynb), [`preprocessing_sunspot.ipynb`](https://github.com/dinesh-k-natarajan/autoregressive-conv-rnn/blob/main/preprocessing_sunspot.ipynb), [`utils/preprocess.py`](https://github.com/dinesh-k-natarajan/autoregressive-conv-rnn/blob/main/utils/preprocess.py)

### 1.2. Data windowing for time series prediction
In order to prepare the dataset for TS prediction, data windowing is necessary. The proposed model uses parallel paths with downsampled input TS to 1/2 and 1/4 of input length. Thus, the `window_size` of the input TS should be a multiple of 4.
 
The `prediction_horizon` depends on the nature of the prediction. For one-step TS predictions, a prediction horizon of 1 is used. For multi-step predictions, prediction horizons = {3,5,7} are used.

The input TS of length = 20 and the target TS of length = {1,3,5,7} form the dataset for supervised learning.

Implementation can be found in: [`preprocessing_temperature.ipynb`](https://github.com/dinesh-k-natarajan/autoregressive-conv-rnn/blob/main/preprocessing_temperature.ipynb), [`preprocessing_sunspot.ipynb`](https://github.com/dinesh-k-natarajan/autoregressive-conv-rnn/blob/main/preprocessing_sunspot.ipynb), [`utils/preprocess.py`](https://github.com/dinesh-k-natarajan/autoregressive-conv-rnn/blob/main/utils/preprocess.py)

## 2. Models

### 2.1. Simple LSTM

A baseline LSTM model is used for comparison to the proposed model. The paper does not describe the architecture of the baseline LSTM model used. In this implementation, a three-layer LSTM followed by an output layer of size `prediction_horizon` is used. 

Implementation can be found in: [`utils/models.py`](https://github.com/dinesh-k-natarajan/autoregressive-conv-rnn/blob/main/utils/models.py)

### 2.2. Autoregressive Convolutional Recurrent Neural Network (ACRNN)

The paper describes the general structure of the proposed model but not the specific architecture used to report the results.
Based on the Section 2 of the paper and improving on [this](https://github.com/KurochkinAlexey/ConvRNN) unofficial Pytorch implementation of CRNN, the proposed model has been implemented here using Tensorflow `2.4.1`.

Implementation can be found in: [`utils/models.py`](https://github.com/dinesh-k-natarajan/autoregressive-conv-rnn/blob/main/utils/models.py)

## 3. Experiments

TWo models were trained and evaluated for one-step and multi-step predictions for the two univariate datasets.
The models were compared using metrics evaluated by k-fold cross validation with k=5. 

The results of the training are saved in the [`trained_models/`](https://github.com/dinesh-k-natarajan/autoregressive-conv-rnn/tree/main/trained_models) directory. This includes the model weights from the best epoch w.r.t minimum validation loss, keras model with the said optimum weights and an evolution plot of the losses during training. These three results are saved for each fold of the k-fold evaluation of the models for each dataset.

For one-step time series prediction, the following metrics are used:  
  * MSE - mean squared error between ground truth and prediction  
  * MAE - mean absolute error between ground truth and prediction  
  
For the multi-step time series prediction, the metric used is:  
  * DTW - Dynamic Time Warping (using [FastDTW](https://pypi.org/project/fastdtw/) implementation)

Helper functions for model training and evaluation can be found in: [`utils/model_functions.py`](https://github.com/dinesh-k-natarajan/autoregressive-conv-rnn/blob/main/utils/model_functions.py)
Results and training history of the models can be found in the notebooks: [`1-step_predictions_temperature.ipynb`](https://github.com/dinesh-k-natarajan/autoregressive-conv-rnn/blob/main/1-step_predictions_temperature.ipynb) and [`1-step_predictions_sunspot.ipynb`](https://github.com/dinesh-k-natarajan/autoregressive-conv-rnn/blob/main/1-step_predictions_sunspot.ipynb)

### 2.1.  One-step prediction

Results from the paper are obtained from Table 1 (Page 4).

Table 1. One-step prediction on Temperature Dataset

| Model Name         |   MSE (x 10^2)      |   MAE (x 10)      | 
| :-----------------:|:-------------------:|:-----------------:|
| Simple LSTM (paper)|1.362 +/- 0.126      |0.9197 +/- 0.0400  |  
| ACRNN (paper)      |1.317 +/- 0.083      |0.9019 +/- 0.0290  |  
| Simple LSTM (mine) |1.651 +/- 0.135      |1.0181 +/- 0.0401  | 
| ACRNN (mine)       |1.564 +/- 0.076      |0.9909 +/- 0.0309  |   

Table 2. One-step prediction on Sunspot Dataset

| Model Name         |   MSE (x 10^2)      |  MAE (x 10)       |
| :-----------------:|:-------------------:|:-----------------:|
| Simple LSTM (paper)|0.564 +/- 0.024      |0.5425 +/- 0.1076  |  
| ACRNN (paper)      |0.501 +/- 0.126      |0.5194 +/- 0.0653  |  
| Simple LSTM (mine) |0.542 +/- 0.024      |0.5480 +/- 0.0171  | 
| ACRNN (mine)       |0.478 +/- 0.070      |0.5029 +/- 0.0329  |


### 2.2.  Multi-step prediction

Comparison of DTW loss values for 3-, 5- and 7-step TS predictions of the two models.

The DTW computation was implemented using [dtaidistance](https://dtaidistance.readthedocs.io/en/latest/) Python package. 
Some clarity is needed regarding the correct computation of DTW loss:  
  * Is DTW loss computed between the entire target TS and predicted TS for the test set (shape = n_data x n_output)?
  * Is DTW loss computed between each sample of the target TS and predicted TS for the test set and then divided by n_samples?
  
The current implementation follows the latter method leading to very low DTW values in comparison to the values reported in the paper. A correct DTW computation is required here.

Table 3. Multi-step prediction on Temperature Dataset

| Model Name         |   3-step            |   5-step          |   7-step          | 
|:------------------:|:-------------------:|:-----------------:|:-----------------:|
| Simple LSTM (paper)|0.592 +/- 0.033      |1.475 +/- 0.143    |2.679 +/- 0.303    |  
| ACRNN (paper)      |0.679 +/- 0.038      |1.672 +/- 0.133    |2.598 +/- 0.118    |  
| Simple LSTM (mine) |0.3870 +/- 0.0041    |0.6661 +/- 0.0366  |0.7999 +/- 0.0557  |
| ACRNN (mine)       |0.3917 +/- 0.0112    |0.6691 +/- 0.0189  |0.8817 +/- 0.0310  |   

Table 4. Multi-step prediction on Sunspot Dataset

| Model Name         |   3-step            |   5-step          |   7-step          | 
| :-----------------:|:-------------------:|:-----------------:|:-----------------:|
| Simple LSTM (paper)|0.317 +/- 0.059      |0.720 +/- 0.111    |1.187 +/- 0.217    |  
| ACRNN (paper)      |0.359 +/- 0.095      |0.859 +/- 0.256    |1.331 +/- 0.362    |  
| Simple LSTM (mine) |0.1971 +/- 0.0154    |0.3216 +/- 0.0093  |0.4151 +/- 0.0114  |
| ACRNN (mine)       |0.1910 +/- 0.0044    |0.3270 +/- 0.0111  |0.4261 +/- 0.0049  | 
 
