import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
from .compile_data import *

# Gets the data with which to build and/or test a new model
def get_data():
    data = pd.read_fwf(os.path.join('dataset', 'dataset.txt'))
    # prep data
    columns = len(data.columns)
    hitColumnsKey = str(columns - 3)
    bounceColumnsKey = str(columns - 2)
    hit = data[hitColumnsKey]
    bounce = data[bounceColumnsKey]
    x = data.drop(bounceColumnsKey, axis=1).drop(hitColumnsKey, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, hit, test_size=0.2, random_state=100)
    return (x_train, x_test, y_train, y_test)

# Build a new model
def build_model(path):
    if os.path.exists(os.path.join(path, 'event_model_v1.joblib')):  
        os.remove(os.path.join(path, 'event_model_v1.joblib'))
    
    x_train, x_test, y_train, y_test = get_data()
    rf = RandomForestClassifier(max_depth=10, random_state=1)
    rf.fit(x_train, y_train)
    joblib.dump(rf, os.path.join(path, 'event_model_v1.joblib'))

# Load a certain model
def load_model(path):
    return joblib.load(os.path.join(path, 'event_model_v1.joblib'))
    
# Display test results for the model that is loaded
def print_test_matrix(path):
    x_train, x_test, y_train, y_test = get_data()
    rf = load_model(path)
    y_rf_test_pred = rf.predict(x_test)
    # get results
    rf_f_score = f1_score(y_test, y_rf_test_pred)
    print("test f score: ", rf_f_score)
    print(confusion_matrix(y_test, y_rf_test_pred))

# If you have a valid dataframe row to make a prediction on, use this    
def predict_from_dataframe(data, path):
    rf = load_model(path)
    return rf.predict(data)

# Make a prediction based on the dictionary format
def predict_from_dict(data, path):
    return predict_from_dataframe(compile_data.compile(data, path))

if (__name__ == "__main__"):
    path = 'trained_model'
    build_model(path)
    print_test_matrix(path)

# TODO: Investigate data augmentation for the model