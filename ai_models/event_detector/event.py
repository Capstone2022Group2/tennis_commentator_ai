import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

# Gets the data with which to build and/or test a new model no_commit/annotations/full_annotations
def get_data():
    data = pd.read_fwf(os.path.join('ai_models/event_detector/dataset', 'dataset.txt'))
    # prep data
    columns = len(data.columns)
    # hitColumnsKey = str(columns - 2)
    # bounceColumnsKey = str(columns - 1)
    hitColumnsKey = str(columns - 3)
    bounceColumnsKey = str(columns - 2)
    hit = data[hitColumnsKey]
    bounce = data[bounceColumnsKey]
    bounce = [int(b) for b in bounce]

    x = data.drop(bounceColumnsKey, axis=1).drop(hitColumnsKey, axis=1)

    #x.reset_index()
    x_train, x_test, y_train, y_test = train_test_split(x, bounce, test_size=0.2, random_state=100)
    return (x_train, x_test, y_train, y_test)

# Build a new model
def build_model():
    x_train, x_test, y_train, y_test = get_data()
    # print(np.any(np.isnan(x_train)))
    # print(np.any(np.isnan(y_train)))
    #y_train = y_train[y_train.isna().any(axis=1)]
    print(y_train)
    # x_train = [float(x) for x in x_train]
    # y_train = [float(y) for y in y_train]
    rf = RandomForestClassifier(max_depth=10, random_state=1)
    rf.fit(x_train, y_train)
    joblib.dump(rf, os.path.join('ai_models/event_detector/trained_model', 'event_model_v2.joblib'))

# Load a certain model
def load_model():
    return joblib.load(os.path.join('ai_models/event_detector/trained_model', 'event_model_v2.joblib'))
    
# Display test results for the model that is loaded
def print_test_matrix():
    x_train, x_test, y_train, y_test = get_data()
    print(x_test)
    rf = load_model()
    y_rf_test_pred = rf.predict(x_test)
    # get results
    rf_f_score = f1_score(y_test, y_rf_test_pred)
    print("test f score: ", rf_f_score)
    print(confusion_matrix(y_test, y_rf_test_pred))
    
def predict(data):
    rf = load_model()
    return rf.predict(data)

if (__name__ == "__main__"):
    print_test_matrix()
    #build_model()

# TODO: Investigate data augmentation for the model