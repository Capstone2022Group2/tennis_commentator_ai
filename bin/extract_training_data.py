import os

import tensorflow as tf
import pandas as pd
import numpy as np
from moviepy.audio.io.AudioFileClip import AudioFileClip

DATADIR = os.path.join('D:\\', 'Documents', 'training_set')
CATEGORIES = ['hits', 'null']

training_data = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for file in os.listdir(path):
        audio_file = AudioFileClip(os.path.join(path, file))
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
            training_data.append([freqs_norm, class_num])

            count += 1
            reader.seek(step_size * count)
            audio = reader.read_chunk(sample_size)

csv_data = []
for row in training_data:
    new_row = []
    new_row.append(row[1])
    new_row.extend(row[0])
    csv_data.append(new_row)

df = pd.DataFrame(csv_data)
df.to_csv('D:\\Documents\\training_set\\training_set.csv', header=False, index=False)
