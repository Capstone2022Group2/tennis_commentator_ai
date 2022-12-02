import os

import numpy as np
from keras import Sequential, Input
from keras.layers import Activation, Dense, Flatten, Conv1D
import tensorflow as tf
from moviepy.audio.io.AudioFileClip import AudioFileClip

data = np.loadtxt('D:\\Documents\\training_set\\training_set.csv', delimiter=',', dtype=float)
training_set = []

for row in data:
    training_set.append([row[1:], int(row[0])])

X = []
y = []

for features, label in training_set:
    X.append(features)
    y.append(label)

X = np.array(X)  #(539, 512)
y = np.array(y)  #(539,)

model = Sequential()
model.add(Input(shape=(512)))
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dense(1, activation=tf.nn.sigmoid))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
model.fit(X, y, batch_size=4, validation_split=0.1, epochs=10)

print(model.predict(tf.expand_dims(X[0], axis=0)))

audio_file = AudioFileClip('..\\trained_models\\test_vid\\test_match1.mp4')

sample_rate = audio_file.fps
sample_size = 512
step_size = 128

C = sample_rate / sample_size
bin_to_freq = np.vectorize(lambda b: round(b * C, 0))
bins = bin_to_freq(np.array(range(0, sample_size)))

reader = audio_file.coreader().reader
reader.seek(0)

audio = reader.read_chunk(sample_size)
count = 0
while audio.size > 0:
    left_channel = audio[:, 0]
    if len(left_channel) < sample_size:
        break
    freqs = np.abs(np.fft.fft(left_channel))
    freqs_norm = np.divide(freqs, freqs.max())
    prediction = model.predict(np.expand_dims(freqs_norm, axis=0))
    print(prediction)
    count += 1
    reader.seek(step_size * count)
    audio = reader.read_chunk(sample_size)

test = 0
