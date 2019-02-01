
# coding: utf-8
# python version 3.7

import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.externals import joblib


#Make sure the drivers file has no repeating ids
def clean_driver_table(drivers):
    d = drivers.driver_id.value_counts()
    ids = d[d > 1].index.tolist()
    if len(ids) > 0 :
        print("*** Driver Ids of drivers whose id appeared more than once ***")
        print(d[d > 1])
        print("Repeating driver Ids removed from drivers table")
        drivers = drivers[~drivers['driver_id'].isin(ids)]
    else:
        print("Drivers table has no repeating ids")
    return drivers

# Combine drivers and test tables into new df
def create_new_test_table(test,drivers):
    test2 = pd.merge(test,drivers,on = 'driver_id')
    test2 = test2.sort_values(['driver_id', 'date'], ascending=[True, True])
    #Convert date to an integer using ordinal
    test2['date'] = pd.to_datetime(test2['date'])
    test2['day_of_week'] = test2.date.dt.dayofweek
    # convert gender to category
    test2['gender'] = test2['gender'].map( {'MALE':1, 'FEMALE':0} )
    # drop unecessary columns in test2
    test2.drop(columns=["date"],inplace=True)
    return test2

# Split into feature and targets so we can scale the features
def split_scale_data(test2):
    #applying scaling without ordering the features will cause different scaling to wrong features
    test2 = test2[['driver_id', 'gender', 'age', 'number_of_kids', 'online_hours', 'day_of_week']]
    test2_features = test2.loc[:, test2.columns != 'online_hours']
    test2_target = test2['online_hours']
    #scale feature
    test2_features = scaler.transform(test2_features)
    print("*** test data format *** ")
    print(test2.head())
    return test2_features, test2_target

# Make prediction and evaluate on RMSE
def predict(test2_features,test2_target):
    print("making predictions . . . ")
    y_pred = RF.predict(test2_features)
    forest_mse = mean_squared_error(y_pred, test2_target)
    forest_rmse = np.sqrt(forest_mse)
    print('Random Forest Regressor RMSE: %.4f' % forest_rmse)

if __name__ == '__main__':
    ## Read data
    drivers = pd.read_csv("../data/drivers.csv")
    test = pd.read_csv("../data/test.csv")
    # Read in scaler
    scaler = joblib.load('../models/scaler.pkl')
    # read in model
    RF = joblib.load('../models/RF2.pkl')

    drivers = clean_driver_table(drivers)
    test2 = create_new_test_table(test,drivers)
    test2_features, test2_target = split_scale_data(test2)
    predict(test2_features,test2_target)



