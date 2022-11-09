import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

data = pd.read_fwf(os.path.join('dataset', 'dataset.txt'))

# prep data
hit = data['20']
bounce = data['21']
x = data.drop('20', axis=1).drop('21', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, hit, test_size=0.2, random_state=100)

# random forest model
rf = RandomForestClassifier(max_depth=10, random_state=1)
rf.fit(x_train, y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

# get results
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
print("train mean squared error: ", rf_train_mse)
print("test mean squared error: ", rf_test_mse)