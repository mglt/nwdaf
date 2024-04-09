import numpy as np
import pandas as pd
pd.options.plotting.backend = "matplotlib"
# Plots
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 10

import time
import datetime
import os
import statistics
import pickle

from IPython.core.display import SVG

# Modeling and Forecasting
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster

# Warnings configuration
import warnings
# warnings.filterwarnings('ignore')
warnings.filterwarnings('once')


def data_subset( cell_id=0, cat_id=0, pe_id=0 ):
  """return a subset of the dataset """
  df = pd.read_csv( 'nwdaf_data.csv' )
  df = df[ ( df.cell_id == cell_id ) & ( df.cat_id == cat_id ) & ( df.pe_id == pe_id ) ] 
  for col_name in ['cell_id', 'cat_id', 'pe_id' ]:
    df = df.drop( col_name, axis=1 )
  df = df.reset_index()    
  return df


def format_data_set( df ):
  """formats the data_set with 'y' and 'date' columns 


[skforefast](https://cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html) requires data being indexed in a column labeled 'date' with datetime objects type with no missing measurements. 

Our 'time' information is a int which we assume to be a number of __seconds__ after an initial datetime - fixed to '2024-01-01'. To generate any missing measurements values we use the [asfreq](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.asfreq.html#pandas.Series.asfreq) function and indicate that measurements are expected every 'second'. 
  """
  time_list = df[ 't' ].to_list()
  freq = time_list[ 1 ] - time_list[ 0 ]
  
  date_list = []
  for t in time_list:
    date_list.append( datetime.datetime.fromisoformat('2024-01-01') +\
                           datetime.timedelta( seconds=t ) )
  df[ 't' ] = date_list
  df = df.rename( columns={ 't' : 'date' } )
  df = df.set_index('date')
  df = df.asfreq(f"{freq}s") # 'ms' 'ns'
  df = df.sort_index()
  df = df.rename( columns={ 'has_anomaly' : 'y' } )
  return df  


def get_train_test( data, test_ratio:int=20, verbose=False ):
  """  split data into train-test
  arg:
    data (df): the data to be split
    test_ratio (int): the percentage of
      data being considered for the testing, that is in
      our case the prediction. Training data will be
      100 - test_ratio
  return:
    data_train (df): the training data frame
    data_test (df): the testing data frame
  """

  test_len = int( test_ratio / 100 * len( data ))

  data_train = data[:-test_len]
  data_test  = data[-test_len:]
  if verbose is True:
    print( f"Spliting data into train-test with a {test_ratio} % test ratio" )
    print(f"Data : {data.index.min()} --- {data.index.max()}  (n={len(data)}) [100%]")
    print(f"  - Training : {data_train.index.min()} --- {data_train.index.max()} (n={len(data_train)}) [{100 - test_ratio}%]")
    print(f"  - Test  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)}) [{test_ratio}%]")
  return  data_train, data_test


def plot_train_test( data, test_ratio:int=20 ):
  
  svg_fn = f"train_test_{test_ratio}.svg"  
  if os.path.isfile( svg_fn ):
    return svg_fn

  data_train, data_test = get_train_test( data, test_ratio=test_ratio )
  fig, ax = plt.subplots(figsize=(6, 2.5))
#  data_train['y'].plot(ax=ax, label='train')
  data_test['y'].plot(ax=ax, label='test')
  ax.legend();
  fig.savefig( svg_fn )
  return svg_fn



# Create and train forecaster

def get_prediction( regressor, data_train, lags, test_len, verbose=False ):
  """Generates predictions for specified regressor / lags,

  args:
    regressor: the regressor being used
    data_train (df): the data frame used for the training
    test_len (int): the size of the predictions, i.e. numbe rof steps to be predicted.
    
  Trains the model and provides test_len predictions.
  """

  
  suffix = f"test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--lag-{lags}--test_len-{test_len}.pickle"  
  pickle_file =  os.path.join( './predictions', f"predictions---{suffix}" )
  if os.path.isfile( pickle_file ):
    with open( pickle_file, 'rb') as f:
      return pickle.load( f )

  forecaster = ForecasterAutoreg(
                 regressor = regressor,
                 lags      = lags # takes the previous 10 seconds
             )
  t_start = time.time()
  forecaster.fit(y=data_train['y'])
  t_stop = time.time()  
  fit_time =  t_stop - t_start
  if verbose is True:
    print( forecaster )
  # Predictions
  t_start = time.time()    
  predictions = forecaster.predict( steps=test_len )
  t_stop = time.time()  
  prediction_time =  t_stop - t_start
  with open( pickle_file, 'wb') as f:
    pickle.dump( ( predictions, fit_time, prediction_time ), f, protocol=pickle.HIGHEST_PROTOCOL)  
  return predictions, fit_time, prediction_time


def predictions_error( data_test, predictions, \
        error_type='mse'\
##     error_type='f1', 
          f1_average = 'binary', # 'micro', 'macro', 'micro' ):
##     error_type='accuracy', 
    """ compares data_test and predictions and retun mean square error """  
  if error_type == 'mse': 
    e = mean_squared_error(
                y_true = data_test['y'],
                y_pred = predictions
            )
  elif error_type == 'f1':
      
    e = sklearn.metrics.f1_score(y_true, y_pred, average=f1_average )
  elif error_type == 'accuracy' :
    e = sklearn.metrics.accuracy_score(y_true, y_pred )
  return e

def predictions_with_threshold( predictions, threshold ):
  """returns the predictions provided by the threshold

  output predictions are 0 or 1.
  """
  return predictions.between( float( threshold ), float( 1 ), inclusive='right' ).map( {True: 1, False: 0} )

def get_min_threshold( data_test, predictions, threshold_step=0.0001, threshold_min=0, threshold_max=1, error_type='mse', f1_average='binary'):
  """ return a threshold value
  arg:
    step (float): the threshold granularity. Values are computed btween 0 and 1.
  return the threshold that provides teh min error. When multiple 
  threshold are found the median value is returned.
  
  """
  threshold_list = np.arange(threshold_min, threshold_max, threshold_step ) 
  threshold_errors = []  
  t_start = time.time()
  for threshold in threshold_list:   
    pred_with_thresh = predictions_with_threshold( predictions, threshold )
    threshold_errors.append( predictions_error( data_test, pred_with_thresh ) )
  t_stop = time.time()
  print( f"Min Treshhold: {t_stop - t_start}s - over {len(threshold_list)} Threshold predictions")
  min_error = min( threshold_errors )
  # getting all indexes corresponding to min error
  df = pd.DataFrame({'col': threshold_errors })
  index_list = df[ df['col'] == min_error ].index.tolist() 
  selected_index = statistics.median( index_list  )  
  return threshold_min + selected_index * threshold_step, min_error 

def lag_evaluation_data( data, regressor, \
        lag_min, lag_max, lag_step, \
        threshold_min=0, threshold_max=1, threshold_step=0.0001,
        test_ratio:int=20):
  """ measures and evaluates predictiosn for multiple lag values.

  Evaluation consists in two set of data:
    1. Measurements - i.e. time and errors -- for various lags.
    2. Predictions 
  """

  ## full data (df) and dictio
  suffix = f"--test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--lag_min-{lag_min}--lag_max-{lag_max}--lag_step-{lag_step}" 
  pickle_file = f"./lag_measurements-{suffix}.pickle"
  if os.path.isfile( pickle_file ) is True:
    return pd.read_pickle( pickle_file ) 

  data_train, data_test = get_train_test( data, test_ratio=test_ratio, verbose=True )  

  test_len = len( data_test )

  predictions_dict = {}
  fit_time_list = []
  prediction_time_list = []
  pred_error_list = []  
  threshold_list = [] 
  threshold_pred_error_list = []
  lag_list = list( range( lag_min, lag_max + lag_step, lag_step ) )
  for lag in lag_list:
    t_start = time.time()  
    predictions, fit_time, prediction_time = get_prediction( regressor, data_train, lag, test_len, verbose=False )
    t_stop = time.time()  

    print( f"  - lag_evaluation {lag} get_prediction executed in {t_stop - t_start}s")  
    predictions_dict[ lag ] = predictions
    fit_time_list.append( fit_time )
    prediction_time_list.append( prediction_time ) 
#    for error_type in [ 'mse', 'f1', 'accuracy' ]:
    pred_error_list.append( predictions_error( data_test, predictions ) )
    ## considering threshold  
    threshold, threshold_pred_error = get_min_threshold( data_test, predictions,\
            threshold_step=threshold_step, threshold_min=threshold_min,\
            threshold_max=threshold_max )  
    threshold_list.append( threshold )  
    threshold_pred_error_list.append ( threshold_pred_error )
    t_stop = time.time()  
    print( f"  - lag {lag} full treatment executed in {t_stop - t_start}s")  
      
  df =  pd.DataFrame( { 'lag' : lag_list, 
#                        'predictions' : predictions_list,
                        'fit_time' : fit_time_list,
                        'prediction_time' : prediction_time_list,
                        'prediction_error' : pred_error_list, 
                        'threshold' : threshold_list, 
                        'threshold_prediction_error' : threshold_pred_error_list } )
  df.to_pickle( pickle_file )
  return df


def plot_lag_evaluation( data, regressor, lag_min, lag_max, lag_step, 
        threshold_min=0, threshold_max=1, threshold_step=0.0001,
        test_ratio:int=20 ):

  suffix = f"--test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--lag_min-{lag_min}--lag_max-{lag_max}--lag_step-{lag_step}"  
  svg_fn = f"./lag_measurements-{suffix}.svg"  

  if os.path.isfile( svg_fn ):
    return svg_fn

  df = lag_evaluation_data( data, regressor, lag_min, lag_max, lag_step, test_ratio=test_ratio)      
  fig, axes = plt.subplots(figsize=(12, 10), nrows=5, ncols=1 ) 
  fig.suptitle( 'Evaluation of Lag Cost / Benefit' )  
  x = df[ 'lag' ].to_list()  
  axes[ 0 ].plot( x, df['fit_time'].to_list())
#  axes[ 0 ].set_title( 'Fit Time (~training time) versus Lag ')  
  axes[ 0 ].set(xlabel='Lags', ylabel='Fit Time (s)')  
  axes[ 1 ].plot( x, df['prediction_time'].to_list())
#  axes[ 1 ].set_title( 'Prediction time versus Lag')  
  axes[ 1 ].set(xlabel='Lags', ylabel='Prediction Time (s)')      
  axes[ 2 ].plot( x, df['prediction_error'].to_list())
#  axes[ 2 ].set_title( 'Mean-Square Error versus Lag')  
  axes[ 2 ].set(xlabel='Lags', ylabel='Prediction Error')          
  axes[ 3 ].plot( x, df['threshold'].to_list())
  axes[ 3 ].set(xlabel='Lags', ylabel='Threshold')          
  axes[ 4 ].plot( x, df['threshold_prediction_error'].to_list())
  axes[ 4 ].set(xlabel='Lags', ylabel='Thresholded Prediction Error')          

#  print( f"plot_lag_evaluation: saving SVG: {svg_fn}" )
  fig.savefig( svg_fn )
  return svg_fn
#  SVG(filename=svg_fn )

def plot_lag_predictions( data, regressor, plot_lag_list, 
        threshold_min=0, threshold_max=1, threshold_step=0.0001, 
        test_ratio:int=20 ):

  svg_fn = f"lag_predictions---test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--plot_lags-{plot_lag_list}.svg" 
  if os.path.isfile( svg_fn ):
    return svg_fn

  data_train, data_test = get_train_test( data, test_ratio=test_ratio )

  fig, axes = plt.subplots(figsize=(7, 3 * len(plot_lag_list)), nrows=len(plot_lag_list), ncols=1 ) 
  fig.suptitle( 'Predictions versus Tests' )
  ax_index = 0  
  for plot_lag in plot_lag_list:
    predictions, fit_time, prediction_time = get_prediction( regressor, data_train, plot_lag, len( data_test ), verbose=False )

    threshold, threshold_pred_error = get_min_threshold( data_test, predictions, threshold_step=threshold_step, threshold_min=threshold_min, threshold_max=threshold_max )  
    y_threshold = predictions_with_threshold( predictions, threshold ) 

    axes[ ax_index ].plot( data_test[ 'y' ], label=f"test" )
    axes[ ax_index ].plot( predictions, label=f"predictions [lag={plot_lag}]" )
    axes[ ax_index ].plot( y_threshold, label=f"threshold predictions [lag={plot_lag}]" )  
    axes[ ax_index ].legend();
    ax_index += 1  
  fig.savefig( svg_fn )
  return svg_fn
#  print( f"plot_lag_predictions: saving SVG: {svg_fn}" )
#  SVG(filename=svg_fn )

if __name__ == "__main__":

  ## working data set
  df = format_data_set( data_subset( cell_id=0, cat_id=0, pe_id=0 ) )

  ## 20 % of measurements will be considered for testing,
  test_ratio = 20
  ## showing training / testing set
  plot_train_test( df, test_ratio=20 )

  ## lag is an important parameter for time series. We will 
  ## estimate the cost / benefit over using a shorter lag value
  ## evcen if non optimal.

  regressor = RandomForestRegressor(\
              random_state=123, 
              n_estimators=100, 
              max_depth=10 )  
  
  lag_min = 10
  lag_max = 2000
  lag_step = 10

  threshold_min = 0
  threshold_max = 1
  threshold_step = 0.001

  plot_lag_evaluation( df, regressor, lag_min, lag_max, lag_step, \
    threshold_min=threshold_min, threshold_max=threshold_max, threshold_step=threshold_step,
    test_ratio=20 )

  plot_lag_list=[10, 500, 1000, 1500, 1736, 2000 ]
  plot_lag_predictions( df, regressor, plot_lag_list, \
    threshold_min=threshold_min, threshold_max=threshold_max, threshold_step=threshold_step,
    test_ratio=20 )
#          threshold_min=, threshold_max=1, threshold_step=0.0001, test_ratio=20 )
