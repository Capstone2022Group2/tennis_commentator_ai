import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import torch
import joblib

data = pd.read_fwf(os.path.join('dataset', 'dataset.txt'))
# prep data
columns = len(data.columns)
hitColumnsKey = str(columns - 3)
bounceColumnsKey = str(columns - 2)
hit = data[hitColumnsKey]
bounce = data[bounceColumnsKey]
x = data.drop(bounceColumnsKey, axis=1).drop(hitColumnsKey, axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, hit, test_size=0.2, random_state=100)

# if we have a trained model, load that
if os.path.exists(os.path.join('trained_model', 'event_model_v1.joblib')):
    rf = joblib.load(os.path.join('trained_model', 'event_model_v1.joblib'))
# if we have a dataset, load that
elif os.path.exists(os.path.join('dataset', 'dataset.txt')):
    # random forest model
    rf = RandomForestClassifier(max_depth=10, random_state=1)
    rf.fit(x_train, y_train)
    joblib.dump(rf, os.path.join('trained_model', 'event_model_v1.joblib'))
else:
    print('No model found!')

y_rf_test_pred = rf.predict(x_test)

# get results
rf_f_score = f1_score(y_test, y_rf_test_pred)
print("test f score: ", rf_f_score)
print(confusion_matrix(y_test, y_rf_test_pred))

# Use f-measure (f-score) instead of RMS
# Investigate data augmentation