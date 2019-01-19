import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('../data/df2.csv', sep=',')
features = tpot_data.drop('hours_online_int', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['hours_online_int'].values, random_state=None)

# Average CV score on the training set was:-9.912086341158101
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    RandomForestRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=5, min_samples_split=12, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
results_int = [int(i) for i in results]
print (results_int[:15])
