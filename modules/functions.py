# Warnings
import warnings
warnings.filterwarnings("ignore")

# Basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import seaborn as sns
from scipy import stats
from functools import reduce

# Statsmodels
import statsmodels.api as sm
import pmdarima as pmd
from pmdarima.arima import auto_arima
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

# Machine Learning models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    precision_score

)

from xgboost import XGBRegressor

def bcrp_dataframe( series , start_date , end_date, freq):
    '''
    Objective:
        This function uses the API interface of the Peruvian Central Bank of Reserve (BCRP) to create a pandas dataframe 
        with time series data available at the BCRP Statistics Database
    
    Input:
        series (str/list) = The code of the series that we will extract from the BCRP
        
        start_date (str)  = The starting date in format "yyyy-mm"
        
        end_date (str)    = The ending date in format "yyyy-mm"

        freq (str)        = The frequency of the series. It can have one the following values: 
                            "Diaria", " Mensual", "Trimestral", "Anual".
    Output:
        It returns a pandas dataframe with a time-series index including the series that we have extracted
         
    '''
    
    url_base = 'https://estadisticas.bcrp.gob.pe/estadisticas/series/api/'
    
    month_s  = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    month_d  = ['-01-','-02-','-03-','-04-','-05-','-06-','-07-','-08-','-09-','-10-','-11-','-12-']

    month_s_mensual  = ['Ene.','Feb.','Mar.','Abr.','May.','Jun.','Jul.','Ago.','Sep.','Oct.','Nov.','Dic.']
    month_d_mensual = ['01-01-','01-02-','01-03-','01-04-','01-05-','01-06-','01-07-','01-08-','01-09-','01-10-','01-11-','01-12-']

    month_s_trimestral = ['T1.','T2.','T3.','T4.']
    month_d_trimestral = ['01-03-','01-06-','01-09-','01-12-']
    
    form_out = '/json'
    
    cod_var = series
    period = '/' + start_date + '/' + end_date
    
    df = pd.DataFrame()
    
    try:

        for j in cod_var:
            url_aux   = url_base + j + form_out + period
            resp      = requests.get(url_aux)
            resp_json = resp.json()
            periods   = resp_json['periods']

            value = []
            dates = []

            for i in periods:
                aux_dat = i['name']
                aux_val = i['values']
                dates.append(aux_dat)
                value.append(float(aux_val[0]))

            dict_aux = {'Fecha' : dates, 
                         resp_json['config']['series'][0]['name'] : value}
            df_aux = pd.DataFrame(dict_aux)
            
            if freq == 'Diario' :

                for (s,d) in zip(month_s,month_d):
                    df_aux['Fecha'] = df_aux['Fecha'].str.replace(s,d)
                df_aux['Fecha'] = pd.to_datetime(df_aux['Fecha'], format="%Y-%m-%d") 
            
            elif freq == 'Mensual' :

                for (s,d) in zip(month_s_mensual,month_d_mensual):
                    df_aux['Fecha'] = df_aux['Fecha'].str.replace(s,d)                    
                df_aux['Fecha'] = pd.to_datetime(df_aux['Fecha'], format="%d-%m-%Y") 

            elif freq == 'Trimestral' :

                for (s,d) in zip(month_s_trimestral,month_d_trimestral):
                    df_aux['Fecha'] = df_aux['Fecha'].str.replace(s,d)
                df_aux['Fecha'] = pd.to_datetime(df_aux['Fecha'], format="%d-%m-%y")            
                      
            
            df_aux.set_index(df_aux['Fecha'], inplace=True)
            df_aux = df_aux.drop(columns=['Fecha'])
            df    = pd.concat([df, df_aux], axis=1)
            
    except Exception as err:
        print(f'There has been an exception: {err}. Try with a different code.')
        
    return df


def get_RMSE( results ):
    '''
    Objective:
        This function receives a dataframe with both Actual and Predicted values and calculates the cummulative RMSE
        for the number of horizons.
        
    Input:
        results (dataframe) = Dataframe with an Actual and Predicted columns
        
    Output:
        A dataframe with the RMSE for each horizons
    
    '''
    RMSE = []
    rmse = []

    for index, row in results.iterrows():
        sqr_err = (row['Actual'] - row['Predicted'])**2
        rmse.append(sqr_err)
        aux_RMSE = np.sqrt(sum(rmse) / len(rmse))
        RMSE.append(aux_RMSE)

    return RMSE

def get_MAPE( results ):
    '''
    Objective:
        This function receives a dataframe with both Actual and Predicted values and calculates the cummulative MAPE
        for the number of horizons.
        
    Input:
        results (dataframe) = Dataframe with an Actual and Predicted columns
        
    Output:
        A dataframe with the MAPE for each horizons
    
    '''
    MAPE = []
    mape = []

    for index, row in results.iterrows():
        per_err = abs((row['Actual'] - row['Predicted']) / row['Actual'])
        mape.append(per_err)
        aux_MAPE = sum(mape) / len(mape)
        MAPE.append(aux_MAPE)

    return MAPE

# def get_adjusted_R2(results):
#     '''
#     Objective:
#         This function receives a dataframe with both Actual and Predicted values and calculates the adjusted R-squared
#         for the predictions in the context of time series.
        
#     Input:
#         results (dataframe) - DataFrame with 'Actual' and 'Prediction' columns
        
#     Output:
#         A single adjusted R-squared value for the predictions
#     '''
#     actual_mean = results['Actual'].mean()
#     n = len(results)
#     k = 1  # Number of predictors (in this case, only one: 'Prediction')

#     ss_total = ((results['Actual'] - actual_mean) ** 2).sum()
#     ss_residual = ((results['Actual'] - results['Prediction']) ** 2).sum()

#     r2 = 1 - (ss_residual / ss_total)
#     adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

#     return adjusted_r2


def get_metrics(results, model='model'):
    '''
    Objective:
        This function receives a dataframe with both Actual and Predicted values and calculates a series of metrics.
        
    Input:
        results (dataframe) = Dataframe with an Actual and Predicted columns
        
        model (str)         = Name of the model. We add it for organizing reasons. 
        
    Output:
        One dataframe for each metric for horizons 1 to 12
    
    '''
    
    RMSE = pd.DataFrame(get_RMSE(results), columns = [f'RMSE_{model}'], index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    
    MAPE = pd.DataFrame(get_MAPE(results), columns = [f'MAPE_{model}'], index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        
    return RMSE, MAPE

def highlight_min(df_results):
    '''
    Objective:
        This function highlights the minimum RMSE in each row of a dataframe.
        
    Input:
        df_results (dataframe) = A dataframe
        
    Output:
        A pandas Style object        
    '''
    
    is_min = df_results == df_results.min()
    return ['background-color: lightgray' if v else '' for v in is_min]


def graph_models(df_results, metric = "RMSE",lim=1.5, colors = None, path=None):
    '''
    Objective:
        This function graphs the errors (RMSE of MAPE) of the different models.
        
    Input:
        df_results (dataframe) = A dataframe
        
        lim (float)            = Indicates the limit of the y-axis

        colors (dict)          = A dictionary with column names as keys and colors as values
        
        path   (str)           = Output path of the figure  
        
    Output:
        A matplotlib.pyplot plot      
    '''
    
    fig, ax = plt.subplots()

    for column in df_results.columns:
        if colors and column in colors:
            ax.plot(df_results.index, df_results[column], label=column, color=colors[column])
        else:
            # Plot with a default color if no color specified for a column
            ax.plot(df_results.index, df_results[column], label=column)
    
    ax.set_xlabel("Horizons")
    ax.set_ylabel(metric + ' as % of the benchmark')
    
    ax.set_ylim(0, lim)
    
    ax.legend()
    
    plt.savefig(path, bbox_inches='tight', dpi=300)
    
    plt.show()

def graph_coefficients(vars_df, value = "Coefficient", path=None):
    '''
    Objective:
        This function graphs the coefficients with their respective score.
        
    Input:
        vars_df (dataframe) = A dataframe
        
        value (str)            = Either "Coefficient" or "Importance Score"
        
    Output:
        A matplotlib.pyplot plot      
    '''

    plt.bar(vars_df["Var"], vars_df[value])
    
    plt.ylabel(value)
    
    plt.tick_params(axis = "x", rotation=90, labelsize=8)
    
    plt.savefig(path, bbox_inches='tight', dpi=300)
    
    plt.show()
    

def get_trend(df, period=12):
    '''
    Objective:
        This function removes the seasonal component of a each column of the timeseries dataframe.
        
    Input:
        df (dataframe)          = A dataframe where each column represents a different time series
        
        period (int)            = The peridiocity of the series. by default it is 12
        
    Output:
        A dataframe where the seasonal component has been eliminated      
    '''
    
    trend_df = pd.DataFrame()

    for col in df.columns:
        stl_result = STL(df[col], period=period).fit()
        trend_df[col] = stl_result.trend

    return trend_df


def lags(data, lag, raw=False):
   
   if raw == True:
       lag_range = range(1, lag)
   else:
       lag_range = range(0, lag)

   # Shift the data 
   data = pd.concat([data.shift(i+1) for i in lag_range],axis=1)

   # Name columns
   data.columns = ['lag_%d' %i for i in lag_range]
    
   # Drop rows with NaN values 
   return data.dropna()


def diebold_mariano_test(y_true, y_pred1, y_pred2, h):
    T = len(y_true) - h
    if T <= 0:
        return np.nan  # Not enough data points to calculate DM statistic
    d = np.zeros(T)
    
    for t in range(T):
        e1 = y_true.iloc[t+h] - y_pred1.iloc[t+h]
        e2 = y_true.iloc[t+h] - y_pred2.iloc[t+h]
        d[t] = (e1 ** 2) - (e2 ** 2)  # Example with squared error

    d_bar = np.mean(d)
    var_d = np.var(d, ddof=1)
    
    DM_statistic = d_bar / np.sqrt(var_d / T)
    return DM_statistic

def calculate_p_value(dm_statistic):
    if np.isnan(dm_statistic):
        return np.nan
    p_value = 2 * (1 - norm.cdf(abs(dm_statistic)))
    return p_value

def create_lagged_features(data, lags):
    lagged_data = pd.concat(
        [data.shift(lag).add_suffix(f'_lag{lag}') for lag in range(1, lags + 1)], axis=1)
    return lagged_data

### DIEBOLD-MARIANO test code
# Author   : John Tsang
# Date     : December 7th, 2017
# Purpose  : Implement the Diebold-Mariano Test (DM test) to compare 
#            forecast accuracy
# Input    : 1) actual_lst: the list of actual values
#            2) pred1_lst : the first list of predicted values
#            3) pred2_lst : the second list of predicted values
#            4) h         : the number of stpes ahead
#            5) crit      : a string specifying the criterion 
#                             i)  MSE : the mean squared error
#                            ii)  MAD : the mean absolute deviation
#                           iii) MAPE : the mean absolute percentage error
#                            iv) poly : use power function to weigh the errors
#            6) poly      : the power for crit power 
#                           (it is only meaningful when crit is "poly")
# Condition: 1) length of actual_lst, pred1_lst and pred2_lst is equal
#            2) h must be an integer and it must be greater than 0 and less than 
#               the length of actual_lst.
#            3) crit must take the 4 values specified in Input
#            4) Each value of actual_lst, pred1_lst and pred2_lst must
#               be numerical values. Missing values will not be accepted.
#            5) power must be a numerical value.
# Return   : a named-tuple of 2 elements
#            1) p_value : the p-value of the DM test
#            2) DM      : the test statistics of the DM test
##########################################################
# References:
#
# Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of 
#   prediction mean squared errors. International Journal of forecasting, 
#   13(2), 281-291.
#
# Diebold, F. X. and Mariano, R. S. (1995), Comparing predictive accuracy, 
#   Journal of business & economic statistics 13(3), 253-264.
#
##########################################################
# Author   : John Tsang
# Date     : December 7th, 2017
# Purpose  : Implement the Diebold-Mariano Test (DM test) to compare 
#            forecast accuracy   

def dm_test(actual_lst, pred1_lst, pred2_lst, h = 1, crit="MSE", power = 2):
 
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_act = len(actual_lst)
        len_p1  = len(pred1_lst)
        len_p2  = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt,msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt,msg)  
        # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")  
        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True
        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            is_actual_ok = compiled_regex(str(abs(actual)))
            is_pred1_ok = compiled_regex(str(abs(pred1)))
            is_pred2_ok = compiled_regex(str(abs(pred2)))
            if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):  
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return (rt,msg)
        return (rt,msg)
    
    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    
    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    
    rt = dm_return(DM = DM_stat, p_value = p_value)
    
    return rt
