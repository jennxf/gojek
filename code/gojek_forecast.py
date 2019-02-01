
# coding: utf-8
#python 3.7

import statsmodels.api as sm
import numpy as np
import itertools
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import operator
import datetime as dt
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

#read in formatted train and test data
def get_data(df_file_path, test_file_path):
    df = pd.read_csv(df_file_path)
    df.index = pd.to_datetime(df.date)
    df.drop(['date','day_of_week_name'],axis=1,inplace=True)

    test2 = pd.read_csv(test_file_path)
    test2.index = pd.to_datetime(test2.date)
    test2.drop('date',axis=1,inplace=True)

    #re-arrange test columns to follow df
    test2= test2[df.columns]
    test2.sort_values(['driver_id', 'date'], ascending=[True, True],inplace=True)
    return df, test2


#concat train and test data
def create_data(df,test2,n_ids):
    driver_id_list = df.driver_id.unique()[:n_ids]
    data = pd.concat([df, test2])
    data = data[data['driver_id'].isin(driver_id_list)]
    return data

#Sarimax model for each driver
#mean_rmse: (sum rmse per driver / total number of driver)
#list_rmse :(all rmse for each driver)
# Overall rmse: for all true_values, and predicted, get the RMSE. regardless of driver id
def sarimax_rmse(data,test2,start_date,end_date):
    true_all = []
    predicted_all = []
    rmse_list = []
    for i in data.driver_id.unique():
        print("For driver id: {0}".format(i))
        df_tmp = data[data.driver_id==i]
        exog_train = df_tmp.loc['2017-06-01':'2017-06-21'].loc[:,df_tmp.columns!='online_hours']
        exog_pred = df_tmp.loc[start:end].loc[:,df_tmp.columns!='online_hours']
        mod = sm.tsa.statespace.SARIMAX(df_tmp.loc['2017-06-01':'2017-06-21']['online_hours'],
                                exog=exog_train,
                                order=(0, 1, 1),
                                seasonal_order=(1, 0, 0, 7),
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                                )
                                
        results = mod.fit(disp=0)
        p = results.get_prediction(start=pd.to_datetime(start), 
                           end= pd.to_datetime(end),
                           exog = exog_pred
                           )
        #convert negative predictions to 0
        predicted = p.predicted_mean.apply(lambda x: 0 if x < 0 else x)
                                        
        test3 = test2[test2.driver_id==i]
        y_true = test3.online_hours
        test3.drop(['online_hours'],axis=1,inplace=True)

        true_all += list(y_true)
        predicted_all += list(predicted)
        rmse = np.sqrt(mean_squared_error(y_true,predicted))
        rmse_list.append(rmse)
        print("mean RMSE for driver {0}: {1}".format(i,rmse))
        print(" Predicted values for driver {0}: {0}".format(i,predicted))
        print(" ")


    overall_rmse = np.sqrt(mean_squared_error(true_all,predicted_all))
    print("** Overall RMSE for all drivers in list: {0} **".format(overall_rmse))
    return np.mean(rmse_list),rmse_list,overall_rmse



if __name__ == '__main__':
    #predict period
    start = '2017-06-22'
    end = '2017-06-28'
    #train test filepath
    df_file_path = '../data/df2.csv'
    test_file_path = '../data/test2.csv'

    df,test2 = get_data(df_file_path,test_file_path)
    data = create_data(df,test2,10)
    mean_rmse, rmse_list, overall_rmse = sarimax_rmse(data,test2,start,end)


