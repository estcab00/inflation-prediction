import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy import stats

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
    
    month_s  = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Set','Oct','Nov','Dic']
    month_d  = ['-01-','-02-','-03-','-04-','-05-','-06-','-07-','-08-','-09-','-10-','-11-','-12-']

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

            elif freq == 'Mensual' :

                for (s,d) in zip(month_s,month_d):
                    df_aux['Fecha'] = df_aux['Fecha'].str.replace(s,d)

            elif freq == 'Trimestral' :

                for (s,d) in zip(month_s_trimestral,month_d_trimestral):
                    df_aux['Fecha'] = df_aux['Fecha'].str.replace(s,d)

            df_aux['Fecha'] = pd.to_datetime(df_aux['Fecha'])            
                      
            
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