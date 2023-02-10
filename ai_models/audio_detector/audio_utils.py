import os

import numpy as np
import pandas as pd
from keras import Sequential, Input
from keras.layers import Dense
from moviepy.audio.io.AudioFileClip import AudioFileClip
import tensorflow as tf


# CONSTANTS
from sklearn.preprocessing import LabelEncoder

CATEGORIES = ['hits', 'null']

# Number of frames of audio used in FFT, which will output this number of frequencies in its result
SAMPLE_SIZE = 512

# How many frames of audio should program skip ahead before taking a new FFT
STEP_SIZE = 32


# Provide path to a directory of subfolders, one folder for each class type
# Subfolders should contain mp3 files for training
def extract_training_data(data_path: str, output_path: str):
    training_data = []
    # Iterate over each class subfolder
    for category in CATEGORIES:
        print('\nExtracting ' + str(category) + ' data...')
        path = os.path.join(data_path, category)
        # Iterate over files in the class subfolder
        training_files = os.listdir(path)
        for file_idx, file in enumerate(training_files):
            print('\rFile ' + str(file_idx + 1) + '/' + str(len(training_files)), end='')
            # Initialize reader to start of audio file and obtain first samlple
            audio_file = AudioFileClip(os.path.join(path, file))
            reader = audio_file.coreader().reader
            reader.seek(0)
            audio = reader.read_chunk(SAMPLE_SIZE)
            count = 0
            # Loop continues as long as a full sample can be obtained
            # Applying FFT to audio sample and normalizing results then adding them to the training data
            while audio.size == 2 * SAMPLE_SIZE:
                left_channel = audio[:, 0]
                freqs = np.abs(np.fft.fft(left_channel))
                freqs_norm = np.divide(freqs, freqs.max())
                new_row = [category]
                new_row.extend(freqs_norm)
                training_data.append(new_row)

                # Get next audio sample
                count += 1
                reader.seek(STEP_SIZE * count)
                audio = reader.read_chunk(SAMPLE_SIZE)

    # Save training data to CSV
    df = pd.DataFrame(training_data)
    output_path = output_path if output_path.endswith('.csv') else output_path + '.csv'
    df.to_csv(output_path, header=False, index=False)


# Trains a model from the provided csv data, returns the model if no output_path is specified
# Model is saved as a directory, no extension should be included
def train_model(csv_path: str, output_path: str):
    # Read in data from the csv produced by extract_training_data, shuffle rows
    data = pd.read_csv(csv_path, sep=',', header=None, keep_default_na=False)
    data = np.array(data)
    np.random.seed(0)
    np.random.shuffle(data)

    # Seperate the labels and features into different variables
    x = np.array(data[:, 1:], dtype=float)
    y = np.array(data[:, 0], dtype=str)

    # Convert classnames into integer values
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    # Define the model
    model = Sequential()
    model.add(Input(shape=512))
    model.add(Dense(512, activation=tf.nn.relu))
    model.add(Dense(512, activation=tf.nn.sigmoid))
    model.add(Dense(512, activation=tf.nn.relu))
    model.add(Dense(2, activation=tf.nn.softmax))

    # Compile and train the model with 30% validation
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='sparse_categorical_accuracy')
    model.fit(x, y, validation_split=0.3, epochs=10, shuffle=False, batch_size=10)

    # Save or return the model
    if output_path is not None:
        model.save(output_path)
    else:
        return model


# Data should be an array of SAMPLE_SIZE floats between 0 and 1 representing intensities of SAMPLE_SIZE frequency bands
def predict(data: list, model):
    result = model.predict(np.expand_dims(data, axis=0), verbose=0).round(0)[0]
    if result[0] == 1:
        return 'HIT'
    elif result[1] == 1:
        return 'NULL'


def main():
    # Example showing predictions being made on 'test_match.mp4' using a model saved as 'trained_model'
    model = tf.keras.models.load_model('./trained_model')
    audio_file = AudioFileClip('D:/Documents/training_set/test_match.mp4')
    reader = audio_file.coreader().reader
    reader.initialize(0)
    reader.seek(0)
    audio = reader.read_chunk(SAMPLE_SIZE)
    count = 0
    while audio.size == 2 * SAMPLE_SIZE:
        left_channel = audio[:, 0]
        freqs = np.abs(np.fft.fft(left_channel))
        freqs_norm = np.divide(freqs, freqs.max())
        print(predict(freqs_norm, model))

        # Using higher step size than for training to run faster
        count += 1
        reader.seek(3 * STEP_SIZE * count)
        audio = reader.read_chunk(SAMPLE_SIZE)


if __name__ == '__main__':
    main()
