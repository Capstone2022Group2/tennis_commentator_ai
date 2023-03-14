import numpy as np
from moviepy.audio.io.AudioFileClip import AudioFileClip
import tensorflow as tf
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip

import ai_models.audio_detector.audio_utils as aud
from matplotlib import pyplot as plt


def main():
    audio_file = AudioFileClip('D:/Documents/training_set/test_match.mp4')
    model = tf.keras.models.load_model('../ai_models/audio_detector/trained_model')
    reader = audio_file.coreader().reader
    reader.seek(0)
    audio = reader.read_chunk(aud.SAMPLE_SIZE)
    count = 0
    STEP_SIZE = 880
    hits = []
    while audio.size == 2 * aud.SAMPLE_SIZE:
        left_channel = audio[:, 0]
        freqs = np.abs(np.fft.fft(left_channel))
        freqs_norm = np.divide(freqs, freqs.max())
        timestamp = count * STEP_SIZE / audio_file.fps
        prediction = aud.predict(freqs_norm, model)
        if prediction[0] > 0.5:
            hits.append(timestamp)

        count += 1
        reader.seek(count * STEP_SIZE)
        audio = reader.read_chunk(aud.SAMPLE_SIZE)

    video = VideoFileClip('D:/Documents/training_set/test_match.mp4')
    clips = [video]
    for hit in hits:
        clip = TextClip('HIT', fontsize=40, color='white')
        clip = clip.set_start(hit).set_position((10, 10)).set_duration(0.1)
        clips.append(clip)

    final = CompositeVideoClip(clips)
    final.write_videofile('D:/Documents/training_set/output.mp4')

if __name__ == '__main__':
    main()
