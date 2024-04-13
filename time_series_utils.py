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
import sklearn
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
  print( df )
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

def get_prediction( regressor, data_train, lags, test_len, verbose=False, data_id='./' ):
  """Generates predictions for specified regressor / lags,

  args:
    regressor: the regressor being used
    data_train (df): the data frame used for the training
    test_len (int): the size of the predictions, i.e. numbe rof steps to be predicted.
    
  Trains the model and provides test_len predictions.
  """

  test_ratio = int( 100 * test_len / ( len( data_train ) + test_len ) ) 
  suffix = f"test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--lag-{lags}--test_len-{test_len}" 
  pickle_dir = os.path.join( data_id, 'predictions' )
  pickle_file =  os.path.join( pickle_dir, f"predictions---{suffix}.pickle" )
  if os.path.isdir( pickle_dir ) is False:
    os.makedirs( pickle_dir )    
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
        error_type='mse', \
##     error_type='f1', 
          f1_average = 'binary' # 'micro', 'macro', 'micro' ):
##     error_type='accuracy',
  ):
  """ compares data_test and predictions and retun mean square error """ 
  y_true = data_test['y']
  y_pred = predictions
  if error_type == 'mse': 
    e = mean_squared_error( y_true, y_pred )
  elif error_type == 'f1':
    e = sklearn.metrics.f1_score( y_true, y_pred, average=f1_average )
  elif error_type == 'accuracy' :
    e = sklearn.metrics.accuracy_score( y_true, y_pred )
  else:
    raise ValueError( f"Unknown error_type : {error_type}" )
  return e

## def lag_evaluation_error_dict( data_test, predictions,\
def predictions_error_dict( data_test, predictions,\
        error_type_list=[ 'mse', 'f1', 'accuracy' ], \
        f1_average_list=[ 'binary', 'micro', 'macro', 'micro' ] ):
  """
  returns the error dictionary
  """

  y_true = data_test['y']
  y_pred = predictions
  error_dict = {} 
#  raise ValueError( f"y_true: {y_true.head()}\n\ny_pred: {y_pred.head()}" ) 
  for error_type in error_type_list:
    if error_type == 'f1':
      for f1_average in f1_average_list :
#        error_dict[ f"{error_type}_{average}" ].append( predictions_error( data_test, predictions, error_type='f1', f1_average=average) )
        error_dict[ f"{error_type}_{average}" ] = sklearn.metrics.f1_score( y_true, y_pred, average=f1_average ) 
    elif error_type == 'accuracy' :  
#      error_dict[ error_type ].append( predictions_error( data_test, predictions, error_type=error_type ) ) 
      error_dict[ f"{error_type}" ] = sklearn.metrics.accuracy_score( y_true, y_pred )
    elif error_type == 'mse' : 
      error_dict[ f"{error_type}" ] = sklearn.metrics.mean_squared_error( y_true, y_pred )    
    else:  
      raise ValueError( f"Unknown error_type : {error_type}" )
  return error_dict

def predictions_with_threshold( predictions, threshold ):
  """returns the predictions provided by the threshold

  output predictions are 0 or 1.
  """
  return predictions.between( float( threshold ), float( 1 ), inclusive='right' ).map( {True: 1, False: 0} )




def get_optimum_threshold( data_test, predictions, threshold_step=0.0001, threshold_min=0, threshold_max=1, error_type='mse', f1_average='binary', data_id=None, prediction_id=None):
  """ return a threshold value
  arg:
    step (float): the threshold granularity. Values are computed btween 0 and 1.
  return the threshold that provides teh min error. When multiple 
  threshold are found the median value is returned.
  
  """
  if data_id is not None and prediction_id is not None:
    suffix = f"data_id-{data_id}--threshold-{threshold_min}-{threshold_max}-{threshold_step}--error_typ-{error_type}"
    if error_type == "f1":
      suffix = f"{suffix}--f1_average-{f1_average}"
    pickle_dir = os.path.join( data_id, 'optimum_threshold' )
    pickle_file =  os.path.join( pickle_dir, f"optimum_threshold---{suffix}.pickle" )
    if os.path.isdir( pickle_dir ) is False:
      os.makedirs( pickle_dir )    
    if os.path.isfile( pickle_file ):
      with open( pickle_file, 'rb') as f:
        return pickle.load( f )
      
#    -regressor-{regressor.__class__.__name__}--lag-{lags}--test_len-{test_len}" 
#  if os.path.isdir( os.path.join( data_id, 'optimum_threshold' )) is False:
#    os.makedirs( pickle_dir )    

  threshold_list = np.arange(threshold_min, threshold_max, threshold_step )
#  print( f"min_max_predictions: \n{predictions}\nmin: {predictions.min()}\nmax: {predictions.max()}" )
  
  threshold_errors = [] 

  predictions_min = predictions.min() - threshold_step
  predictions_max = predictions.max() + threshold_step

  t_start = time.time()
  for threshold in threshold_list: 
    if threshold < predictions_min :
      threshold_min += threshold_step
      continue
    elif  threshold > predictions_max:
      break
    pred_with_thresh = predictions_with_threshold( predictions, threshold )
    threshold_errors.append( predictions_error( data_test, pred_with_thresh, error_type=error_type, f1_average=f1_average ) )
  t_stop = time.time()
  if error_type in [ 'mse' ]:
    min_error = min( threshold_errors )
  elif error_type in [ 'f1', 'accuracy' ]:
    min_error = max( threshold_errors )
  else: 
    raise ValueError( f"Unable to determine the optimum threshold" )    
  # getting all indexes corresponding to min error
  df = pd.DataFrame({'col': threshold_errors })
  index_list = df[ df['col'] == min_error ].index.tolist() 
  selected_index = statistics.median( index_list  ) 
  threshold = threshold_min + selected_index * threshold_step
  if data_id is not None and prediction_id is not None:
    with open( pickle_file, 'wb') as f:
      pickle.dump( ( threshold, min_error ), f, protocol=pickle.HIGHEST_PROTOCOL)  
  print( f"  - Min Treshhold: {t_stop - t_start}s - over {len(threshold_list)} Threshold predictions: {pickle_file }")
  return threshold, min_error 

def get_optimum_threshold_dict( data_test, predictions,\
      threshold_step=0.0001, threshold_min=0, threshold_max=1, 
      error_type_list=[ 'mse', 'f1', 'accuracy' ], \
      f1_average_list=[ 'binary', 'micro', 'macro', 'micro' ], data_id=None, prediction_id=None ):

  threshold_dict = {}
  min_error_dict = {}
  for error_type in error_type_list:
    if error_type == 'f1':
      for average in f1_average_list :
        threshold, min_error = get_optimum_threshold( data_test, predictions, threshold_step=threshold_step, threshold_min=threshold_min, threshold_max=threshold_max, error_type=error_type, f1_average=average, data_id=data_id, prediction_id=prediction_id )
        threshold_dict[ f"{error_type}_{average}" ] = threshold  
        min_error_dict[ f"{error_type}_{average}" ] = min_error 
    elif error_type in [ 'accuracy', 'mse' ] :  
      threshold, min_error = get_optimum_threshold( data_test, predictions, threshold_step=threshold_step, threshold_min=threshold_min, threshold_max=threshold_max, error_type=error_type, data_id=data_id, prediction_id=prediction_id )
      threshold_dict[ f"{error_type}" ] = threshold  
      min_error_dict[ f"{error_type}" ] = min_error 
    else:  
      raise ValueError( f"Unknown error_type : {error_type}" )
  return threshold_dict, min_error_dict


def get_error_label_list( \
        error_type_list=[ 'mse', 'f1', 'accuracy' ],\
        f1_average_list=[ 'binary', 'micro', 'macro', 'micro' ] )-> list:
  """ return the list labels associated to errors/ scores """

  error_label = []
  for error_type in error_type_list :
    if error_type == 'f1':
      for average in f1_average_list :
        error_label.append( f"{error_type}_{average}" )    
    else: 
      error_label.append( f"{error_type}" )
  return error_label

def lag_evaluation_data( data, regressor, \
        lag_min, lag_max, lag_step, \
        threshold_min=0, threshold_max=1, threshold_step=0.0001,
        test_ratio:int=20, \
        error_type_list=[ 'mse', 'f1', 'accuracy' ],\
        f1_average_list=[ 'binary', 'micro', 'macro', 'micro' ],\
        data_id='./' ):

  """ measures and evaluates predictiosn for multiple lag values.

  Evaluation consists in two set of data:
    1. Measurements - i.e. time and errors -- for various lags.
    2. Predictions 
  """

  ## full data (df) and dictio
##  suffix = f"test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--lag_min-{lag_min}--lag_max-{lag_max}--lag_step-{lag_step}" 
  suffix = f"test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--lag-{lag_min}-{lag_max}-{lag_step}--threshold-{threshold_min}-{threshold_max}-{threshold_step}--error_type-{error_type_list}--f1_average-{f1_average_list}" 
#  pickle_file = f"./lag_measurements-{suffix}.pickle"
  pickle_file =  os.path.join( data_id , f"lag_evaluation---{suffix}.pickle" )

  if os.path.isfile( pickle_file ) is True:
    return pd.read_pickle( pickle_file )

  data_train, data_test = get_train_test( data, test_ratio=test_ratio, verbose=True )  

  test_len = len( data_test )

  ## initializing the data dictionary
  threshold_error_label_list = get_error_label_list( error_type_list=error_type_list, f1_average_list=f1_average_list )
#  raise ValueError( f"threshold_error_label_list: {threshold_error_label_list}" )
  ## some metrics like f1/accuraccy are only able to quantify 
  ## the exact match. In our case it only works for th threshold
  ## prediction not the continuous predictions.
  continuous_error_type_list = error_type_list[:]  
  for score_metric in [ 'f1', 'accuracy' ]: 
    if score_metric in error_type_list:
      continuous_error_type_list.remove( score_metric )

  continuous_error_label_list = get_error_label_list( \
          error_type_list=continuous_error_type_list, f1_average_list=[] )

  key_list = [ 'lag', 'fit_time',  'prediction_time' ]
  key_list.extend( [ f"prediction_error_{i}" for i in continuous_error_label_list ] )
  key_list.extend( [ f"optimum_threshold_{i}" for i in threshold_error_label_list ] )
  key_list.extend( [ f"optimum_threshold_error_{i}" for i in threshold_error_label_list ] )

  data_dict = {}
  for k in key_list:
    data_dict[ k ] = []
  lag_list = list( range( lag_min, lag_max + lag_step, lag_step ) )
  data_dict[ 'lag' ] = lag_list

  for lag in lag_list:
    t_start = time.time()  
    predictions, fit_time, prediction_time = get_prediction( regressor, data_train, lag, test_len, verbose=False, data_id=data_id )
    t_stop = time.time()  

    print( f"  - lag_evaluation {lag} get_prediction executed in {t_stop - t_start}s : {pickle_file}")  
#    predictions_dict[ lag ] = predictions
#    fit_time_list.append( fit_time )
    data_dict[ 'fit_time' ].append( fit_time )
#    prediction_time_list.append( prediction_time ) 
    data_dict[ 'prediction_time' ].append( prediction_time )
#    prediction_time_list.append( prediction_time ) 

    ## collecting the errors / scores from the predictions
    ## removing F1 as it only applied on threshold
#    new_error_type_list = error_type_list[:]  
#    for score_metric in [ 'f1', 'accuracy' ]: 
#      if score_metric in error_type_list:
#        new_error_type_list.remove( score_metric )
    error_dict = predictions_error_dict( data_test, predictions,
        error_type_list=continuous_error_type_list, f1_average_list=[] )
#    print( f"error_dict: {error_dict}")
    for k in continuous_error_label_list:
      data_dict[ f"prediction_error_{k}" ].append( error_dict[ k ] )    
#    for error_type in [ 'mse', 'f1', 'accuracy' ]:
#        axes[ 2 ].plot( x, df[ error_type ], label=error_type )
#    pred_error_list.append( predictions_error( data_test, predictions ) )
    ## considering threshold  
#    threshold, threshold_pred_error = get_min_threshold( data_test, predictions,\
#            threshold_step=threshold_step, threshold_min=threshold_min,\
#            threshold_max=threshold_max )  
#    threshold_list.append( threshold )  
#    threshold_error_dict = predictions_error_dict( data_test, predictions )

    ### Handling threshold (only applicable for labeled data) 

    ## collecting the threshold that minimizes the error 
    ## the corresponding error / score is returned.
    ## suffix = f"test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--lag-{lag_min}-{lag_max}-{lag_step}--threshold-{threshold_min}-{threshold_max}-{threshold_step}--error_type-{error_type_list}--f1_average-{f1_average_list}" 
    prediction_id = f"prediction--test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--lag-{lag}" 
    threshold_dict, min_error_dict = get_optimum_threshold_dict( \
            data_test, predictions,\
            threshold_step=threshold_step, \
            threshold_min=threshold_min, \
            threshold_max=threshold_max,\
            error_type_list=error_type_list,\
            f1_average_list=f1_average_list,\
            data_id=data_id, prediction_id=prediction_id )
    for k in threshold_error_label_list :
      data_dict[ f"optimum_threshold_{k}" ].append( threshold_dict[ k ] )   
#    raise ValueError( f"threshold_dict: {threshold_dict}" )   
    for k in threshold_error_label_list :
      data_dict[ f"optimum_threshold_error_{k}" ].append( min_error_dict[ k ] ) 
#    threshold_pred_error_list.append ( threshold_pred_error )
    t_stop = time.time()  
  print( f"  - lag {lag} full treatment executed in {t_stop - t_start}s") 
#  for k in data_dict.keys():
#    print( f"  - {k}: {len( data_dict[ k ] )}" )    
  df = pd.DataFrame( data_dict )  

##  df =  pd.DataFrame( { 'lag' : lag_list, 
###                        'predictions' : predictions_list,
##                        'fit_time' : fit_time_list,
##                        'prediction_time' : prediction_time_list,
##                        'prediction_error' : pred_error_list, 
##                        'threshold' : threshold_list, 
##                        'threshold_prediction_error' : threshold_pred_error_list } )
  df.to_pickle( pickle_file )
  return df


def plot_lag_evaluation( data, regressor, lag_min, lag_max, lag_step, 
        threshold_min=0, threshold_max=1, threshold_step=0.0001,
        test_ratio:int=20,\
        error_type_list=[ 'mse', 'f1', 'accuracy' ],\
        f1_average_list=[ 'binary', 'micro', 'macro', 'micro' ], \
        data_id='./'):

#  suffix = f"--test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--lag_min-{lag_min}--lag_max-{lag_max}--lag_step-{lag_step}"  
  suffix = f"test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--lag-{lag_min}-{lag_max}-{lag_step}--threshold-{threshold_min}-{threshold_max}-{threshold_step}--error_type-{error_type_list}--f1_average-{f1_average_list}" 
#  svg_fn = f"./lag_measurements-{suffix}.svg"  
  svg_fn =  os.path.join( data_id , f"lag_evaluation---{suffix}.svg" )

  if os.path.isfile( svg_fn ):
    return svg_fn

  threshold_error_label_list = get_error_label_list( error_type_list=error_type_list, f1_average_list=f1_average_list )
#  error_label_list = get_error_label_list( )
  continuous_error_type_list = error_type_list[:]  
  for score_metric in [ 'f1', 'accuracy' ]: 
    if score_metric in error_type_list:
      continuous_error_type_list.remove( score_metric )

  continuous_error_label_list = get_error_label_list( \
          error_type_list=continuous_error_type_list, f1_average_list=[] )

  df = lag_evaluation_data( data, regressor, \
          lag_min, lag_max, lag_step, \
          test_ratio=test_ratio, \
          threshold_min=threshold_min, threshold_max=threshold_max,\
          threshold_step=threshold_step,\
          error_type_list=error_type_list,\
          f1_average_list=f1_average_list, \
          data_id=data_id)
#  print( f"lag_evaluation_data: {df.head( 10 )}") 
  fig, axes = plt.subplots( figsize=(16, 10), nrows=4, ncols=1 ) 
  fig.suptitle( 'Evaluation of Lag Cost / Benefit' )  
  x = df[ 'lag' ].to_list()  
#  axes[ 0 ].plot( x, df['fit_time'].to_list(), label='Fit')
  axes[ 0 ].plot( x, df['fit_time'], label='Fit')
  axes[ 0 ].plot( df['prediction_time'], label='Prediction')
  axes[ 0 ].set_title( 'Fit / Predict Time versus Lag')  
  axes[ 0 ].legend()
  axes[ 0 ].set(xlabel='Lags', ylabel= 'Time (s)')  
#  axes[ 1 ].set_title( 'Prediction time versus Lag')  
#  axes[ 0 ].set(xlabel='Lags', ylabel='Prediction Time (s)') 
#  new_error_type_list = error_type_list[:]  
#  for score_metric in [ 'f1', 'accuracy' ]: 
#    if score_metric in error_type_list:
#      new_error_type_list.remove( score_metric )
  for error_label in continuous_error_label_list:
    axes[ 1 ].plot( x, df[ f"prediction_error_{error_label}" ], label=error_label )
  axes[ 1 ].set(xlabel='Lags', ylabel='Prediction Score / Error')          
  axes[ 1 ].set_title( 'Prediction Score / Error versus Lag')  
  axes[ 1 ].legend()
        
#  for error_type in [ 'mse', 'f1', 'accuracy' ]:
#    if error_type == 'f1':
#      for average in [ 'binary', 'micro', 'macro', 'micro' ] : 
#        axes[ 2 ].plot( x, df[ f"{error_type}_{average}" ], label=error_type )
#    else:  
#      axes[ 2 ].plot( x, df[ error_type ], label=error_type )
#  axes[ 2 ].set_title( 'Mean-Square Error versus Lag')  
  for error_label in threshold_error_label_list :
    axes[ 2 ].plot( x, df[ f"optimum_threshold_{error_label}" ], label=error_label )
#    print( f"plot_lag_measurements: optimum_threshold_{error_label} : {df[ f'optimum_threshold_{error_label}' ].to_list()}") 

  axes[ 2 ].set(xlabel='Lags', ylabel='Optimum Threshold')          
  axes[ 2 ].set_title( 'Optimum Threshold versus Lag')  
  axes[ 2 ].legend()
  for error_label in threshold_error_label_list :
    axes[ 3 ].plot( x, df[ f"optimum_threshold_error_{error_label}" ], label=error_label )
  axes[ 3 ].set(xlabel='Lags', ylabel='Thresholded Prediction Error')          
  axes[ 3 ].set_title( 'Score / Error for Optimum Threshold versus Lag')  
  axes[ 3 ].legend()
  plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.4)  

#  print( f"plot_lag_evaluation: saving SVG: {svg_fn}" )
  fig.savefig( svg_fn )
  return svg_fn
#  SVG(filename=svg_fn )

def plot_lag_predictions( data, regressor, lag_list, 
        threshold_min=0, threshold_max=1, threshold_step=0.0001, 
        test_ratio:int=20, \
        error_type_list=[ 'mse', 'f1', 'accuracy' ],\
        f1_average_list=[ 'binary', 'macro', 'micro' ], data_id=None ):

  suffix = f"test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--plot_lags-{lag_list}--threshold-{threshold_min}-{threshold_max}-{threshold_step}--error_type_list-{error_type_list}--f1_average-{f1_average_list}"
    
  svg_fn = os.path.join( data_id, f"lag_predictions---{suffix}.svg" ) 

  if os.path.isfile( svg_fn ):
    return svg_fn
#  suffix = f"test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--lag-{lag_min}-{lag_max}-{lag_step}--threshold-{threshold_min}-{threshold_max}-{threshold_step}--error_type-{error_type}--f1_average-{f1_average}" 

  data_train, data_test = get_train_test( data, test_ratio=test_ratio )

  error_label_list = get_error_label_list(error_type_list=error_type_list,\
          f1_average_list=f1_average_list )

  fig, axes = plt.subplots(figsize=(16, 3 * len(lag_list)), nrows=len(lag_list), ncols=1 ) 
  fig.suptitle( 'Predictions versus Tests' )
  ax_index = 0  
  for plot_lag in lag_list:
    predictions, fit_time, prediction_time = get_prediction( regressor, data_train, plot_lag, len( data_test ), verbose=False, data_id=data_id )

#    threshold, threshold_pred_error = get_min_threshold( data_test, predictions, threshold_step=threshold_step, threshold_min=threshold_min, threshold_max=threshold_max )
    prediction_id = f"prediction--test_ratio_{test_ratio}--regressor-{regressor.__class__.__name__}--lag-{plot_lag}" 

    threshold_dict, min_error_dict = get_optimum_threshold_dict( \
            data_test, predictions,\
            threshold_step=threshold_step, \
            threshold_min=threshold_min, \
            threshold_max=threshold_max,\
            error_type_list=error_type_list,\
            f1_average_list=f1_average_list,\
            data_id=data_id, prediction_id=prediction_id )
#    print( f"threshold_dict: {threshold_dict.keys()}")
    axes[ ax_index ].plot( data_test[ 'y' ], label=f"Test" )
    axes[ ax_index ].plot( predictions, label=f"Predictions [lag={plot_lag}]" )
    for error_label in error_label_list :
      y_threshold = predictions_with_threshold( predictions,\
              threshold_dict[ error_label ] ) 
      axes[ ax_index ].plot( y_threshold, label=f"Threshold [{error_label}]" )  
#      print( f"plot_lag_predictions: {error_label} : {y_threshold.to_list()[ :20 ] }") 
    axes[ ax_index ].set_title( f"Lag: {plot_lag}" )
    axes[ ax_index ].legend()
    ax_index += 1 
  plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.4)  
  fig.savefig( svg_fn )
  return svg_fn
#  print( f"plot_lag_predictions: saving SVG: {svg_fn}" )
#  SVG(filename=svg_fn )

def traffic_evaluation( cell_id, cat_id, pe_id ):
  df = format_data_set( data_subset( cell_id=cell_id, cat_id=cat_id, pe_id=pe_id ) )
  data_id = f"cell_id-{cell_id}--cat_id-{cat_id}--pe_id-{pe_id}"
  print( f"data_id: {data_id}" )
  if os.path.isdir( data_id ) is False:
    os.mkdir( data_id )    
  ## 20 % of measurements will be considered for testing,
  test_ratio = 20


  ## showing training / testing set
  plot_train_test( df, test_ratio=test_ratio )

  ## lag is an important parameter for time series. We will 
  ## estimate the cost / benefit over using a shorter lag value
  ## evcen if non optimal.

  regressor = RandomForestRegressor(\
              random_state=123, 
              n_estimators=100, 
              max_depth=10 )  
  
  lag_min = 10
  lag_max = 500
  lag_step = 10

  threshold_min = 0
  threshold_max = 1
  threshold_step = 0.001

  error_type_list=[ 'mse', 'f1', 'accuracy' ]
  f1_average_list=[ 'binary', 'macro', 'micro' ]
#  f1_average_list=[ 'micro', 'macro', 'micro' ]

  plot_lag_evaluation( df, regressor, lag_min, lag_max, lag_step, \
    threshold_min=threshold_min, threshold_max=threshold_max, threshold_step=threshold_step,
    error_type_list=error_type_list,\
    f1_average_list=f1_average_list,\
    test_ratio=test_ratio,
    data_id=data_id )

#  plot_lag_list=[10, 500, 1000, 1500, 1736, 2000 ]
  lag_list=[10, 100, 500, 1000 ]
  plot_lag_predictions( df, regressor, lag_list, \
    threshold_min=threshold_min, threshold_max=threshold_max, threshold_step=threshold_step,
    error_type_list=error_type_list,\
    f1_average_list=f1_average_list,\
    test_ratio=test_ratio,
    data_id=data_id )

if __name__ == "__main__":
  
#  cell_id = 4 
#  for cat_id in [ 0, 1, 2 ]:
#    for pe_id in [ 0, 1, 2, 3, 4]:   
#      traffic_evaluation( cell_id, cat_id, pe_id )
  cat_id = 2
  for pe_id in [ 3, 4 ]:
    for cell_id in [ 3, 4 ]:
      traffic_evaluation( cell_id, cat_id, pe_id )
          
#  for cell_id in  [ 0, 1, 2, 3, 4]:
#    traffic_evaluation( cell_id, 0, 0 )

