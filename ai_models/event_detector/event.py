import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
import joblib
from scipy.stats import randint
# from keras import models
# from keras.layers import Dense, Dropout
import tensorflow as tf
from tensorflow import keras
#import compile_data
#from .compile_data import *

model_name = 'replay_detection.joblib'
dataset_name = 'labeled_replay_data.txt'

def data_summary(X_train, y_train, X_test, y_test):
    """Summarize current state of dataset"""
    print('Train images shape:', X_train.shape)
    print('Train labels shape:', y_train.shape)
    print('Test images shape:', X_test.shape)
    print('Test labels shape:', y_test.shape)
    print('Train labels:', y_train)
    print('Test labels:', y_test)

# Gets the data with which to build and/or test a new model no_commit/annotations/full_annotations
def get_data():
    print(f'{dataset_name}')
    data = pd.read_fwf(os.path.join('no_commit/annotations/replay_annotations', dataset_name))
    #data = pd.read_fwf(os.path.join('ai_models/event_detector/dataset', 'dataset.txt'))

    # prep data
    columns = len(data.columns)
    #hitColumnsKey = str(columns - 2)
    bounceColumnsKey = str(columns - 1)
    #hitColumnsKey = str(columns - 2)
    # bounceColumnsKey = str(columns - 1)
    #labels = data[[hitColumnsKey, bounceColumnsKey]]
    #labels = [int(label) for label in labels]
    bounce = data[bounceColumnsKey]
    #bounce = [int(b) for b in bounce]
    #print(bounce)
    x = data.drop(bounceColumnsKey, axis=1)#.drop(hitColumnsKey, axis=1)
    x.reset_index()
    #print(x)
    x_train, x_test, y_train, y_test = train_test_split(x, bounce, test_size=0.2, random_state=100, shuffle=True)
    #print()
    return (x_train, x_test, y_train, y_test)

# Build a new model
def build_model(path):
    if os.path.exists(os.path.join(path, model_name)):  
        os.remove(os.path.join(path, model_name ))
    
    x_train, x_test, y_train, y_test = get_data()
    rf = RandomForestClassifier(max_depth=15, n_estimators=410, random_state=39)

    rf.fit(x_train, y_train)
    joblib.dump(rf, os.path.join(path, model_name ))

# Load a certain model
def load_model(path):
    return joblib.load(os.path.join(path, model_name))
    #return keras.models.load_model("ai_models/event_detector/trained_model/nn2")
    
# Display test results for the model that is loaded
def print_test_matrix(path, x_test, y_test):
    #x_train, x_test, y_train, y_test = get_data()
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
    return predict_from_dataframe(compile(data, path))


def tune_hyperparameters(path):
    x_train, x_test, y_train, y_test = get_data()
    param_dist = {'n_estimators': randint(50,100),
              'max_depth': randint(1,10),
              'random_state': randint(1,50)}
    # # param_dist = {'n_estimators': randint(50,500),
    # #           'learning_rate': randint(1,20),
    # #           'random_state': randint(1,50)}

    # # param_dist = {'random_state': randint(1,50),
    # #           'min_samples_split': randint(2,10)}
    # param_dist = {'random_state': randint(1,50)}

    # # Create a random forest classifier
    rf = RandomForestClassifier()
    # #rf = AdaBoostClassifier()
    # #rf = svm.SVC()
    # #rf = tree.DecisionTreeClassifier()
    # #rf = LogisticRegression()

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=5, 
                                    cv=5)

    # # Reshape data
    # x_train = x_train.reshape((x_train.shape[0], 1 * 20))
    # # x_train = x_train.astype('float32') / 255
    # x_test = x_test.reshape((x_test.shape[0], 20))
    # # x_test = x_test.astype('float32') / 255

    # # Categorically encode labels
    # y_train = keras.utils.to_categorical(y_train, 3)
    # y_test = keras.utils.to_categorical(y_test, 3)

    # # data_summary(x_train, y_train, x_test, y_test)

    # model = keras.models.Sequential()
    # # model.add(keras.layers.Dense(512,activation = 'relu',input_shape=(12,)))
    # # model.add(keras.layers.Dense(128,activation = 'relu'))                           
    # # model.add(keras.layers.Dense(2,activation = 'softmax'))
    # model.add(keras.layers.Dense(10,activation = 'relu',input_shape=(12,)))
    # #model.add(keras.layers.Dropout(0.5, noise_shape = None, seed = None))
    # model.add(keras.layers.Dense(8,activation = 'relu'))
    # #model.add(keras.layers.Dropout(0.5, noise_shape = None, seed = None))
    # model.add(keras.layers.Dense(6,activation = 'relu'))
    # model.add(keras.layers.Dropout(0.5, noise_shape = None, seed = None))                               
    # model.add(keras.layers.Dense(3,activation = 'softmax'))
    # model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics =['categorical_accuracy'])

    # model.fit(x_train.values, y_train,
    #       epochs=1300,
    #       validation_data=(x_test.values, y_test))
    
    # score = model.evaluate(x_test.values, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    # model.save('ai_models/event_detector/trained_model/nn2')
    # Fit the random search object to the data
    rand_search.fit(x_train, y_train)

    best_rf = rand_search.best_estimator_
    joblib.dump(best_rf, os.path.join(path, model_name ))

    # Print the best hyperparameters
    print('Best hyperparameters:',  rand_search.best_params_)

    print_test_matrix('ai_models/event_detector/trained_model', x_test, y_test)

if (__name__ == "__main__"):
    # path = 'trained_model'
    #build_model('ai_models/event_detector/trained_model')
    # datasets = ['new_label_3_output.txt', 'new_label_hit_bounce.txt', 'new_label_no_hit.txt',
    #             'new_labels2_3_output.txt', 'new_labels2_hit_bounce.txt', 'new_labels2_no_hit.txt',
    #             'new_w_players_3_output.txt', 'new_w_players_hit_bounce.txt', 'new_w_players_no_hit.txt',
    #             'new2_w_players_3_output.txt', 'new2_w_players_hit_bounce.txt', 'new2_w_players_no_hit.txt']
    
    # for dataset in datasets:
    #     dataset_name = dataset
    #     print('-------------------------------------------------')
        
    #     tune_hyperparameters('ai_models/event_detector/trained_model')
    tune_hyperparameters('ai_models/event_detector/trained_model')
    #print_test_matrix('ai_models/event_detector/trained_model')
    #get_data()

# TODO: Investigate data augmentation for the model