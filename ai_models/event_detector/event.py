import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
from scipy.stats import randint
#import compile_data

model_name = 'event_model_old.joblib'

# Gets the data with which to build and/or test a new model no_commit/annotations/full_annotations
def get_data():
    data = pd.read_fwf(os.path.join('no_commit/annotations/3_frame', 'dataset3.txt'))
    #data = pd.read_fwf(os.path.join('ai_models/event_detector/dataset', 'dataset.txt'))

    # prep data
    columns = len(data.columns)
    #hitColumnsKey = str(columns - 2)
    bounceColumnsKey = str(columns - 1)
    hitColumnsKey = str(columns - 2)
    # bounceColumnsKey = str(columns - 1)
    #labels = data[[hitColumnsKey, bounceColumnsKey]]
    #labels = [int(label) for label in labels]
    bounce = data[bounceColumnsKey]
    #bounce = [int(b) for b in bounce]
    #print(data[hitColumnsKey])
    x = data.drop(bounceColumnsKey, axis=1).drop(hitColumnsKey, axis=1)
    x.reset_index()
    #print(x)
    x_train, x_test, y_train, y_test = train_test_split(x, bounce, test_size=0.2, random_state=100)
    return (x_train, x_test, y_train, y_test)

# Build a new model
def build_model(path):
    if os.path.exists(os.path.join(path, model_name)):  
        os.remove(os.path.join(path, model_name ))
    
    x_train, x_test, y_train, y_test = get_data()
    #rf = RandomForestClassifier(max_depth=12, n_estimators=140, random_state=27)
    rf = RandomForestClassifier(max_depth=19, n_estimators=336, random_state=41)

    rf.fit(x_train, y_train)
    joblib.dump(rf, os.path.join(path, model_name ))

# Load a certain model
def load_model(path):
    return joblib.load(os.path.join(path, model_name))
    
# Display test results for the model that is loaded
def print_test_matrix(path):
    x_train, x_test, y_train, y_test = get_data()
    rf = load_model(path)
    y_rf_test_pred = rf.predict(x_test)
    # get results
    rf_f_score = f1_score(y_test, y_rf_test_pred, average='weighted')
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_rf_test_pred)
    print("test f score: ", rf_f_score)
    print("accuracy: " + str(acc) )
    print(confusion_matrix(y_test, y_rf_test_pred))

# If you have a valid dataframe row to make a prediction on, use this    
def predict_from_dataframe(data, path):
    rf = load_model(path)
    return rf.predict(data)

# Make a prediction based on the dictionary format
def predict_from_dict(data, path):
    return predict_from_dataframe(compile_data.compile(data, path))


def tune_hyperparameters():
    x_train, x_test, y_train, y_test = get_data()
    param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20),
              'random_state': randint(1,50)}

    # Create a random forest classifier
    rf = RandomForestClassifier()

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=5, 
                                    cv=5)

    # Fit the random search object to the data
    rand_search.fit(x_train, y_train)

    # best_rf = rand_search.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:',  rand_search.best_params_)

if (__name__ == "__main__"):
    # path = 'trained_model'
    #build_model('ai_models/event_detector/trained_model')
    tune_hyperparameters()
    #print_test_matrix('ai_models/event_detector/trained_model')
    #get_data()

# TODO: Investigate data augmentation for the model