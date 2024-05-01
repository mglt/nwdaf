import numpy as np
import pandas as pd
pd.options.plotting.backend = "matplotlib"
#pd.options.plotting.backend = "plotly"
# Plots
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 10

import plotly.express as px
import time
import datetime
import os
import statistics
import pickle
import seaborn as sns
sns.set(style='white')

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import skforecast
import skforecast.ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster

#https://stackoverflow.com/questions/51452569/how-to-resize-rescale-a-svg-graphic-in-an-ipython-jupyter-notebook
from IPython.display import SVG, display, HTML
import base64
_html_template='<img width="{}" src="data:image/svg+xml;base64,{}" >'

def show_svg(svg_file, width="100%"):
  svg = SVG( svg_file ).data.encode("utf8")  
  b64 = base64.b64encode(svg).decode("utf8") 
  text = f'<img width="{width}" src="data:image/svg+xml;base64,{b64}" >'
  return HTML(text)

# Warnings configuration
import warnings
# warnings.filterwarnings('ignore')
warnings.filterwarnings('once')

class NWDAFDataSet:

  def __init__( self, cell_id, cat_id, pe_id):
    self.df = self.format_data_set( self.data_subset( cell_id, cat_id, pe_id ) )
    self.id = f"cell_id-{cell_id}--cat_id-{cat_id}--pe_id-{pe_id}"


  def data_subset( self,  cell_id, cat_id, pe_id ):
    """return a subset of the dataset """
    df = pd.read_csv( 'nwdaf_data.csv' )
    df = df[ ( df.cell_id == cell_id ) & ( df.cat_id == cat_id ) & ( df.pe_id == pe_id ) ] 
    for col_name in ['cell_id', 'cat_id', 'pe_id' ]:
      df = df.drop( col_name, axis=1 )
    self.df = df.reset_index()    
    return df

  def format_data_set( self, df ):
    """formats the data_set with 'y' and 'date' columns 


[skforefast](https://cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html) requires data being indexed in a column labeled 'date' with datetime objects type with no missing measurements. 

Our 'time' information is a int which we assume to be a number of __seconds__ after an initial datetime - fixed to '2024-01-01'. To generate any missing measurements values we use the [asfreq](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.asfreq.html#pandas.Series.asfreq) function and indicate that measurements are expected every 'second'. 
    """
#    print( df )
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
    df = df.rename( columns={ 'load' : 'exog_1' } )
    return df  


  def get_train_test( self, train_len:int, verbose=False ):
    """  split data into train-test
    arg:
      train_len (int): the percentage of
        data being considered for the testing, that is in
        our case the prediction. Training data will be
        100 - test_ratio
    return:
      data_train (df): the training data frame
      data_test (df): the testing data frame
    """
     
    data_train = self.df[:train_len]
    data_test  = self.df[-( len( self.df ) - train_len ):]
    if verbose is True:
      print( f"Spliting data into train-test with train_len={train_len}" )
      print(f"Data : {self.df.index.min()} --- {self.df.index.max()}  (n={len(self.df)})")
      print(f"  - Training : {data_train.index.min()} --- {data_train.index.max()} (n={len(data_train)}) [{len(data_train)/len(df):.2f}%]")
      print(f"  - Test  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)}) [{len(data_test)/len(df):.2f}%]")
    return  data_train, data_test


  def plot_train_test( self, train_len:int ):
  
    svg_fn = f"train_test_{train_len}.svg"  
    if os.path.isfile( svg_fn ):
      return svg_fn

    data_train, data_test = self.get_train_test( train_len=train_len )
    fig, ax = plt.subplots(figsize=(6, 2.5))
    data_test['y'].plot(ax=ax, label='test')
    ax.legend();
    fig.savefig( svg_fn )
    return svg_fn


class TimeSerieLagAnalysis:

  def __init__( self, data:NWDAFDataSet, train_len, regressor, 
          forecaster='ForecasterAutoreg'
             ):   

    self.data = data
    self.train_len = train_len
    self.forecaster = forecaster 
    self.data_train, self.data_test = self.data.get_train_test( train_len )
    
    self.data.plot_train_test( train_len )

    self.regressor = regressor 

    self.id = f"train_len_{self.train_len}--forecaster-{forecaster}--regressor-{self.regressor.__class__.__name__}-" 
    self.output_dir = self.data.id
    print( f"Storing results in {self.output_dir}" )
    if os.path.isdir( self.output_dir ) is False:
      os.mkdir( self.output_dir )    


  def get_prediction( self, lag, verbose=False ):
    """Generates predictions for specified regressor / lags,

    args:
      regressor: the regressor being used
      data_train (df): the data frame used for the training
      test_len (int): the size of the predictions, i.e. numbe rof steps to be predicted.
      
    Trains the model and provides test_len predictions.
    """
    steps = len( self.data_test )
    pickle_dir = os.path.join( self.output_dir, 'predictions' )
    suffix = f"{self.id}--lag-{lag}"
    pickle_file = os.path.join( pickle_dir, f"predictions--{suffix}.pickle" )
    if os.path.isdir( pickle_dir ) is False:
      os.makedirs( pickle_dir )    
    if os.path.isfile( pickle_file ):
      with open( pickle_file, 'rb') as f:
        return pickle.load( f )
    if self.forecaster in [ 'ForecasterAutoreg', 'ForecasterAutoregWithExog' ]:
      forecaster = skforecast.ForecasterAutoreg.ForecasterAutoreg(
                   regressor = self.regressor,
                   lags      = lag )
    else:
      raise ValueError( f"Unknown forecaster {self.forecaster}" )    
    t_start = time.time()
    if self.forecaster in [ 'ForecasterAutoreg' ]:
      forecaster.fit(y=self.data_train['y'])
    elif self.forecaster == 'ForecasterAutoregWithExog':
      forecaster.fit( y=self.data_train['y'], exog=self.data_train['exog_1'] )
    else:
      raise ValueError( f"Unknown forecaster {self.forecaster}" )    
    t_stop = time.time()  
    fit_time =  t_stop - t_start
    if verbose is True:
      print( forecaster )
    # Predictions
    t_start = time.time()    
    if self.forecaster in [ 'ForecasterAutoreg' ]:
      predictions = forecaster.predict( steps=steps )
    elif self.forecaster == 'ForecasterAutoregWithExog':
      predictions = forecaster.predict( steps=steps, exog=self.data_test['exog_1'] )
    else:
      raise ValueError( f"Unknown forecaster {self.forecaster}" )    
    t_stop = time.time()  
    prediction_time =  t_stop - t_start
    with open( pickle_file, 'wb') as f:
      pickle.dump( ( predictions, fit_time, prediction_time, forecaster ), f, protocol=pickle.HIGHEST_PROTOCOL)  
    return predictions, fit_time, prediction_time, forecaster


  def get_optimum_threshold( self, predictions, metric='f1', \
        prediction_id=None, threshold_min=0, threshold_max=1, threshold_step=0.001 ):
    """ return the threshold value to optimizes the metric

    arg:
      step (float): the threshold granularity. Values are computed btween 0 and 1.
    return the threshold that provides teh min error. When multiple 
    threshold are found the median value is returned.
  
    protection_id provides some context to the get_optimum_function. That is to prevent that the get_optimum function called in different mixed up with the values. 
    In our case, we need to charaacterize sufficiently the predictions. 
    The predictions is characterized by f"{self.class_id}--test_len-{test_len}--lag-{lags}". self.class_id is provided by the optimum function so lag and test_len are the only argument needed. 
    """

    if prediction_id is not None:
      ## checking prediction_id
      label_list = [ "lag" ]
      for label in label_list:
        if label not in prediction_id:
          raise ValueError( f"WARNING: prediction_id {prediction_id} "\
                 f"does not seem to contain {label}.\n"\
                 f"It is strongly recommended that label id contains "\
                 f"information related to {label_list}. ") 
      pickle_dir = os.path.join( self.output_dir, 'optimum_threshold' )
      threshold_id = f"threshold-{threshold_min}-{threshold_max}-{threshold_step}"
      suffix = f"{self.id}--{prediction_id}--{threshold_id}--metric-{metric}"

      pickle_file =  os.path.join( pickle_dir, f"optimum_threshold--{suffix}.pickle" )
      if os.path.isdir( pickle_dir ) is False:
        os.makedirs( pickle_dir )    
      if os.path.isfile( pickle_file ):
        print( f"pickle_file: {pickle_file}" )  
        with open( pickle_file, 'rb') as f:
          return pickle.load( f )
        
    
    threshold_list = np.arange( threshold_min, threshold_max, threshold_step )
    threshold_errors = [] 

    predictions_min = predictions.min() - threshold_step
    predictions_max = predictions.max() + threshold_step

    y_true = self.data_test['y']

    t_start = time.time()
    for threshold in threshold_list: 
      if threshold < predictions_min :
        threshold_min += threshold_step
        continue
      elif  threshold > predictions_max:
        break
      ## building a thresholded_prediction  
      y_pred = predictions.between( float( threshold ), float( 1 ), inclusive='right' ).map( {True: 1, False: 0} )
      if metric == 'f1_binary':
        score = sklearn.metrics.f1_score( y_true, y_pred, average='binary', pos_label=1)
      elif metric == 'f1_micro':
        score = sklearn.metrics.f1_score( y_true, y_pred, average='micro' )
      elif metric == 'accuracy' :
        score = sklearn.metrics.accuracy_score( y_true, y_pred )
      elif metric == 'precision_binary' :
        score = sklearn.metrics.precision_score(y_true, y_pred, average='binary', pos_label=1 )
      elif metric == 'precision_micro' :
        score = sklearn.metrics.precision_score(y_true, y_pred, average='micro' )
      elif metric == 'average_precision_micro' :
        score = sklearn.metrics.average_precision_score( y_true, y_pred, average='micro' )  
      elif metric == 'recall_binary' :
        score = sklearn.metrics.recall_score(y_true, y_pred, pos_label=1, average='binary' )
      elif metric == 'recall_micro' :
        score = sklearn.metrics.recall_score(y_true, y_pred, average='micro' )
      else:
        raise ValueError( f"Unknown metric {metric}" )     

      threshold_errors.append( score ) 
    t_stop = time.time()
    max_score = max( threshold_errors ) 
    # getting all indexes corresponding to min error
    df = pd.DataFrame({'col': threshold_errors })
    selected_index = statistics.median( df[ df['col'] == max_score ].index.tolist()  ) 
    threshold = threshold_min + selected_index * threshold_step
    if prediction_id is not None:
      with open( pickle_file, 'wb') as f:
        pickle.dump( ( threshold, max_score ), f, protocol=pickle.HIGHEST_PROTOCOL)  
    print( f"  - Min Treshhold: {t_stop - t_start}s - over {len(threshold_list)} Threshold predictions: {pickle_file }")
    return threshold, max_score 


  def plot_lag_forcaster_evaluation( self, lag_min, lag_max, lag_step, test_step:int=None ):
    """ returns measurements to estimate the model (without threshold) """ 
    lag_list = list( range( lag_min, lag_max + lag_step, lag_step ) )

    file_name = f"forcaster_eval--{self.id}--lag-{lag_min}-{lag_max}-{lag_step}"

    pickle_file =  os.path.join( self.output_dir , f"{file_name}.pickle" )
    if os.path.isfile( pickle_file ) is True:
      df = pd.read_pickle( pickle_file )
    else: 
      data_dict = {}
      for m in [ 'fit_time', 'pred_time', 'roc_auc', 'pr_auc', 
                 'mean_sqrt', 'mean_abs', 'r2_score', 'step_nbr' ]:
        data_dict[ m ] = []
      data_dict[ 'lag' ] = lag_list
      y_true = self.data_test['y']

      for lag in lag_list:
        t_start = time.time()  
        y_pred, fit_time, pred_time, _ = self.get_prediction( lag )
        t_stop = time.time()  
        print( f"  - lag_evaluation {lag} get_prediction executed in {t_stop - t_start}s")  
        data_dict[ 'fit_time' ].append( fit_time )
        data_dict[ 'pred_time' ].append( pred_time )
        data_dict[ 'roc_auc' ].append( sklearn.metrics.roc_auc_score( y_true, y_pred) )
        precision, recall, threshold = sklearn.metrics.precision_recall_curve(y_true, y_pred )  
        data_dict[ 'pr_auc' ].append( sklearn.metrics.auc( recall, precision ) )
        data_dict[ 'mean_sqrt' ].append( sklearn.metrics.mean_squared_error( y_true, y_pred ) )
        data_dict[ 'mean_abs' ].append( sklearn.metrics.mean_absolute_error( y_true, y_pred ) )
        data_dict[ 'r2_score' ].append( sklearn.metrics.r2_score( y_true, y_pred)  )
        ## computing the number of step where prediction occurs
        ly_pred = y_pred.to_list()
        ly_pred.reverse()
        y_ref = y_pred[ -1 ]
        step_nbr = len( y_true )
#        print( f"### lag: {lag} ly_true: {ly_pred[-100:]}" )
        for y in ly_pred:
          if y != y_ref :
            break  
          step_nbr -= 1
#          print( f"  -- lag {lag}:  step_nbr: {step_nbr}" )
        data_dict[ 'step_nbr' ].append( step_nbr )    
        t_stop = time.time()  
      print( f"  - lag {lag} full treatment executed in {t_stop - t_start}s") 
      df = pd.DataFrame( data_dict )  
      df.to_pickle( pickle_file )


    svg_fn =  os.path.join( self.output_dir , f"{file_name}.svg" )
    if os.path.isfile( svg_fn ) is False:
      if test_step is not None:
        df = df[ : test_step ]    

      fig, axes = plt.subplots( figsize=(16, 10), nrows=4, ncols=1 ) 
      fig.suptitle( 'Evaluation of Lag Cost / Benefit' )  
      x = df[ 'lag' ].to_list()  
      axes[ 0 ].plot( x, df['fit_time'], label='Fit')
      axes[ 0 ].plot( x, df['pred_time'], label='Prediction')
      axes[ 0 ].set_title( 'Fit / Predict Time versus Lag')  
      axes[ 0 ].legend()
      axes[ 0 ].set(xlabel='Lags', ylabel= 'Time (s)')  
      axes[ 1 ].plot( x, df[ f"roc_auc" ], label="ROC AUC" )
      axes[ 1 ].plot( x, df[ f"pr_auc" ], label="PR AUC" )
      axes[ 1 ].set(xlabel='Lags', ylabel='AUC')          
      axes[ 1 ].set_title( 'ROC AUC and Precision Recall AUC versus Lag')  
      axes[ 1 ].legend()
      axes[ 2 ].plot( x, df[ f"mean_sqrt" ], label="Mean Square Error" )
      axes[ 2 ].plot( x, df[ f"mean_abs" ], label="Mean Absolute Error" )
      axes[ 2 ].plot( x, df[ f"r2_score" ], label="R2 Score" )
      axes[ 2 ].set(xlabel='Lags', ylabel='Error')          
      axes[ 2 ].set_title( 'Error')  
      axes[ 2 ].legend()
      axes[ 3 ].plot( x, df[ f"step_nbr" ] )
      axes[ 3 ].set(xlabel='Lags', ylabel='Steps')          
      axes[ 3 ].set_title( 'Number of Predicted Steps')  
      plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.4)  

      fig.savefig( svg_fn )
    return df, svg_fn

  #### functions intended to select the apprpriated lag 
  #### from the df returned by plot_lag_forcaster_evaluation
  def mMm( self, df, label ):
    """return mean value, and L∞ of the column 'label' """  
    df = df[ label ]
    return df.mean(), df.max() - df.min()
  
  def print_mMm( self, df, label )-> str:
    """return string displaying mean and L∞"""  
    m, l = self.mMm( df, label )  
    return f" ~={m:.2f}, L∞={l:.2f}"
  
  ## finding lag_0, and the smallest possible L∞ for
  
  def optimal_auc_interval( self, df, delta_lag ):
    """returns PR AUC L∞ for delta_lag interval """   
  
    if delta_lag %2 == 1:
      raise ValueError( f"Please select even number of delta_lag  {delta_lag}" )   
    auc_delta = {}    
    for lag in range( int( df[ 'lag' ].min() + delta_lag / 2 ), int( df[ 'lag' ].max() - delta_lag / 2 ) ) :
      if lag not in df[ 'lag' ].values:
        continue    
      sub_df = df[ ( df[ 'lag' ] >= lag - delta_lag / 2 ) & ( df['lag'] <= lag + delta_lag / 2 ) ]
      mean, delta = self.mMm( sub_df, 'pr_auc' )
      auc_delta[ lag ] = delta
    auc_delta_list =  sorted( auc_delta.items(), key=lambda x: x[1])
    ## graphical representation  
    fig = px.line( x = [ i[1] for i in auc_delta_list ], y= [ i[0] for i in auc_delta_list ] ) 
    fig.update_layout( 
      title=f'Center of the {delta_lag} lag intervall versus  L∞ PR AUC',
      xaxis_title='L∞ PR AUC',
      yaxis_title=f'{delta_lag} lag interval center')
    fig.show()
    ## providing information about optimal lag and correpsonding PR AUC 
    print( f"L∞ AUC for {delta_lag} lag interval: {auc_delta_list[ : 20 ]}..." )
    optimum_lag = auc_delta_list[ 0 ][ 0 ]
    print( f"  - optimum lag: {optimum_lag}" ) 
    sub_df = df[ ( df[ 'lag' ] >= optimum_lag - delta_lag / 2  ) & ( df['lag'] <= optimum_lag + delta_lag / 2 ) ]
    print( f"  - optimum PR AUC: {self.print_mMm(sub_df, 'pr_auc')}" ) 
    return auc_delta_list   
    
  def lag_matching_fit_time( self, df_auc, auc_delta_list, max_fit_time ):
    """ check that the PR AUC optimized interval provides a training time < max_fit_time"""   
    for i in auc_delta_list :
      if float( df_auc.loc[ df_auc[ 'lag' ] == i[0] ][ 'fit_time' ].iloc[ 0 ] ) < max_fit_time :
        return  i[ 0 ]   
      
  def print_model_info( self, df_auc, lag, delta_lag ):
    """ Characterizes our model """    
    print( f"Information for model with lag: {lag} +/- {delta_lag}" )
    sub_df = df_auc[ ( df_auc[ 'lag' ] >= lag - delta_lag / 2  ) & ( df_auc['lag'] <= lag + delta_lag / 2 ) ]          
    ## Prediction time
    print( f"  - predition time (s): {self.print_mMm( sub_df, 'pred_time' )}" )
    ## Fit time
#    import sklearn.linear_model
    regr = sklearn.linear_model.LinearRegression()
    regr.fit( sub_df[[ 'lag' ]], sub_df[[ 'fit_time' ]])
    print(f"  - fit time (s): {self.print_mMm( sub_df, 'fit_time' )} -- Coefficient {regr.coef_} s/lag" )
    ## AUC PR
    print( f"  - PR AUC: {self.print_mMm(sub_df, 'pr_auc')}" )
    ## Errors:
    print( f"  - Mean Square Error : {self.print_mMm(sub_df, 'mean_sqrt')}" )
    print( f"  - Mean Absolute Error : {self.print_mMm(sub_df, 'mean_abs')}" )
    print( f"  - R2 Score : {self.print_mMm(sub_df, 'r2_score')}" )
    ## Number of steps:
    print( f"  - steps Prediction : {sub_df[ 'step_nbr' ].min():.2f} steps" )
  


  def select_lag_from_forcaster_evaluation( self, df_auc, delta_lag:int, 
          max_fit_time:int=None, lag_min:int=None, lag_max:int=None ):
    """ selects the proper lag value

    Args:
      df_auc: the dataframe output by plot_lag_forcaster_evaluation
      delta_lag: the lag interval over which we perform the measurements. This interval is used to abstract as much as possible the findings from a local artefact. 
      max_fit_time: the maximum time in seconds that fit time is permitted. This is to avoid that model that requires too long training time be selected. 
      lag_min, lag_max: lag windows that defines the acceptable lag values that we consider acceptable. These values MUST be included in df_auc.

    The selected_lag is found as the value that optimizes L∞ for PR AUC. This works fine as long as the PR AUC varoes over a constant value. If that is not the case, for example, if PR AUC decreases, we need to refine the proper lag intervall.  
    """
    ## get lag that optimize PR AUC L∞
    auc_delta_list = self.optimal_auc_interval( df_auc,  delta_lag)

    ## Ensuring or get the best lag that also match the fit time constrain
    if max_fit_time is not None:
      selected_lag = self.lag_matching_fit_time( df_auc, auc_delta_list, max_fit_time )
    print(f"  - selected_lag: {selected_lag} +/- {delta_lag} -- optimizing PR AUC with fit time < {max_fit_time}" )

    self.print_model_info( df_auc, selected_lag, delta_lag )
    return selected_lag 


  def plot_thresholded_forecaster_evaluation( self, lag_min, lag_max, lag_step, 
          threshold_min=0, threshold_max=1, threshold_step=0.001, 
          test_step:int=None ):

    lag_list = list( range( lag_min, lag_max + lag_step, lag_step ) )
    lag_id = f"lag-{lag_min}-{lag_max}-{lag_step}"
    threshold_list = np.arange( threshold_min, threshold_max, threshold_step )
    threshold_id = f"threshold-{threshold_min}-{threshold_max}-{threshold_step}"
    file_name = f"threshold_forecaster--{self.id}--{lag_id}--{threshold_id}"

    metric_list = [ 'f1_binary', 'f1_micro', 'accuracy', \
            'precision_binary', 'precision_micro',\
            'average_precision_micro',  'recall_binary', 'recall_micro' ]

    pickle_file =  os.path.join( self.output_dir , f"{file_name}.pickle" )
    if os.path.isfile( pickle_file ) is True:
      df = pd.read_pickle( pickle_file )
    else: 
      data_dict = {}
      for metric in  metric_list:
        data_dict[ f"threshold_{metric}" ] = []
        data_dict[ f"score_{metric}" ] = []
      data_dict[ 'lag' ] = lag_list
    
      for lag in lag_list:
        t_start = time.time()  
        predictions, fit_time, prediction_time, forecaster = \
                self.get_prediction( lag )
        t_stop = time.time()  
    
        prediction_id = f"thresholded_prediction--lag-{lag}"
        for metric in metric_list:
          threshold, min_error = self.get_optimum_threshold( predictions, 
                  metric=metric, prediction_id=prediction_id, 
                  threshold_min=threshold_min, threshold_max=threshold_max,\
                          threshold_step=threshold_step )
          data_dict[ f"threshold_{metric}" ].append( threshold )   
          data_dict[ f"score_{metric}" ].append( min_error ) 
        t_stop = time.time()  
      print( f"  - lag {lag} full treatment executed in {t_stop - t_start}s") 
      df = pd.DataFrame( data_dict )  
      df.to_pickle( pickle_file )
    
    svg_fn =  os.path.join( self.output_dir , f"{file_name}.svg" )
    if os.path.isfile( svg_fn ) is False:
      if test_step is not None:
        df = df[ : test_step ]    

      fig, axes = plt.subplots( figsize=(16, 10), nrows=2, ncols=1 ) 
      fig.suptitle( 'Thresholded Prediction Score versus Lab' )  
      x = df[ 'lag' ].to_list() 
      for metric in metric_list:
        axes[ 0 ].plot( x, df[ f"score_{metric}" ], label=metric)
      axes[ 0 ].set_title( 'Optimum Score versus Lag')  
      axes[ 0 ].legend( loc='center left', bbox_to_anchor=(1, 0.5) )
      axes[ 0 ].set(xlabel='Lags', ylabel= 'Score')  
      for metric in metric_list:
        axes[ 1 ].plot( x, df[ f"threshold_{metric}" ], label=metric)
      axes[ 1 ].set_title( 'Optimum Threshold versus Lag')  
      axes[ 1 ].legend( loc='center left', bbox_to_anchor=(1, 0.5) )
      axes[ 1 ].set(xlabel='Lags', ylabel= 'Threshold')  
      plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.4)  
      fig.savefig( svg_fn )
    return df, svg_fn

 #### Functions analyzing the thresholded prediction  


  def select_scores( self, sub_df ):
    score_dict = {} 
    for metric in [ 'f1_binary', 'f1_micro', 'accuracy', 'precision_binary', 'precision_micro', 'average_precision_micro', 'recall_binary', 'recall_micro' ] :
      t_label = f"threshold_{metric}" 
      t_m, t_l = self.mMm( sub_df, t_label) 
      s_label = f"score_{metric}"
      s_m, s_l = self.mMm( sub_df, s_label)
      score_dict[ metric ] = { 'threshold' : [t_m, t_l], 'score' : [s_m, s_l ] }
    return score_dict
  
  def print_mMm_scores( self, sub_df, threshold_var=1, score_var=1 ):
      
    score_dict = self.select_scores( sub_df )
    txt_reject = ""  
    print( f"\nAccepted Metrics:" )
    for metric in  score_dict.keys() : 
      t_label =  f"threshold_{metric}"
      s_label =  f"score_{metric}" 
      t_l = score_dict[ metric ][ 'threshold' ][ 1 ]
      s_l = score_dict[ metric ][ 'score' ][ 1 ]  
      if t_l >= threshold_var or  s_l >= score_var:
        txt_reject +=  f"  - {metric}: [Score: {self.print_mMm( sub_df, s_label)}][Threshold: {self.print_mMm( sub_df, t_label)}]\n" 
      else:
  #      t_label = f"threshold_{metric}"  
        print( f"  - {metric}:[Score: {self.print_mMm( sub_df, s_label)}][Threshold: {self.print_mMm( sub_df, t_label)}]" )
    print( f"\nRejected Metrics (Too high variation):\n{txt_reject}" )
  



  def plot_lag_predictions( self, lag_list, metric_list=[], test_len=None ):

#    metric_label_list = self.get_metric_label_list( 'error_and_score' ) 
    threshold_id = f"threshold-{threshold_min}-{threshold_max}-{threshold_step}"
    metric_list = [ 'f1_binary', 'f1_micro', 'accuracy', \
            'precision_binary','precision_micro', \
            'average_precision_micro',  'recall_binary',\
            'recall_micro' ]
    suffix = f"{self.class_id}--plot_lags-{lag_list}--{threshold_id}--{metric_list}"
      
    svg_fn = os.path.join( self.output_dir, f"lag_predictions--{suffix}.svg" ) 

    if os.path.isfile( svg_fn ):
      return svg_fn

    fig, axes = plt.subplots(figsize=(16, 3 * len(lag_list)), nrows=len(lag_list), ncols=1 ) 
    fig.suptitle( 'Predictions versus Tests' )
    ax_index = 0  
    for plot_lag in lag_list:
      predictions, fit_time, prediction_time = self.get_prediction( plot_lag )

      if test_len is None:
        test_len = len( self.data_test )
      prediction_id = f"prediction--test_len-{test_len}--lag-{plot_lag}" 

      threshold_dict, min_error_dict = self.get_optimum_threshold_dict( 
              predictions, prediction_id=prediction_id )
      axes[ ax_index ].plot( self.data_test[ 'y' ], label=f"Test" )
      axes[ ax_index ].plot( predictions, label=f"Predictions [lag={plot_lag}]" )
      for metric in metric_label_list :
        y_threshold = self.predictions_with_threshold( predictions,\
                threshold_dict[ metric ] ) 
        axes[ ax_index ].plot( y_threshold, label=f"Threshold [{metric}]" )  
      axes[ ax_index ].set_title( f"Lag: {plot_lag}" )
      axes[ ax_index ].legend()
      ax_index += 1 
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.4)  
    fig.savefig( svg_fn )
    return svg_fn




if __name__ == "__main__":
  ### Analysing (Forecaster, regressor) for different lag values 
  SELECT_LAG = True
  if SELECT_LAG is True:
    cell_id=0
    cat_id=0
    pe_id=0
    
    # formating the dataset
    data = NWDAFDataSet ( cell_id=cell_id, cat_id=cat_id, pe_id=pe_id )
    train_len = int( 80 * len( data.df ) / 100 )
    test_len = len( data.df ) - train_len
    
    print( f"test_len : {test_len}" )
    ### select the regressor 
    regressor = sklearn.ensemble.RandomForestRegressor(\
                      random_state=123,
                      n_estimators=100,
                      max_depth=10 )
#    regressor = sklearn.linear_model.Ridge(random_state=123) 
    ## select the forcaster
    forecaster = 'ForecasterAutoregWithExog'
    forecaster = 'ForecasterAutoreg'

    forecaster_eval = TimeSerieLagAnalysis( data, train_len, regressor, forecaster=forecaster )
    lag_min = 2
    lag_max = 2000
    lag_step = 2
    ## plot and computes predictions to measure model AUC for various lag
    df_auc, svg = forecaster_eval.plot_lag_forcaster_evaluation( lag_min, lag_max, lag_step )
    ## select the most appropriated lag
    delta_lag = 100
    max_fit_time = 60
    selected_lag = forecaster_eval.select_lag_from_forcaster_evaluation( df_auc, delta_lag=100, max_fit_time=60)
    ## plot the scored for the thresholded forecaster
    ## we may refine the interval
    lag_min = 2
    lag_max = 800
    lag_step = 2
    df_score, svg = forecaster_eval.plot_thresholded_forecaster_evaluation( lag_min, lag_max, lag_step, 
            threshold_min=0, threshold_max=1, threshold_step=0.001, 
            test_step=None )
    sub_df_score = df_score[ ( df_score[ 'lag' ] >= selected_lag - delta_lag ) & ( df_score['lag'] <= selected_lag + delta_lag ) ]
  print_mMm_scores( sub_df_score, threshold_var=0.25, score_var=0.25 )

