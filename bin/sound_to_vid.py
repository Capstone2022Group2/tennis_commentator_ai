# Example usage
# python sound_to_vid.py -v input.mp4 -o output.mp4 (Optional: -s 16384)

import glob
import math
import os.path
import getopt
import sys

import numpy as np

import matplotlib.pyplot as plot

from moviepy.editor import VideoFileClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

temp_path = './sound_graph_temp'


def create_temp_path():
    os.mkdir(temp_path)


def delete_temp_path():
    if os.path.exists(temp_path):
        for filename in glob.glob(os.path.join(temp_path, '*')):
            os.remove(filename)
        os.rmdir(temp_path)


try:
    opts, args = getopt.getopt(sys.argv[1:], "v:o:s:")
except getopt.GetoptError:
    print('Unexpected error')
    sys.exit(2)

video_path = None
output_path = None

if len(opts) == 0:
    print('Required arguments are -v (input video) and -o (output video), optional arguments are -s (sample size, '
          'default 8192)')
    exit(0)

sample_size = 8192
for opt, arg in opts:
    if opt == '-v':
        video_path = arg
    elif opt == '-o':
        output_path = arg
    elif opt == '-s':
        sample_size = int(arg)

if video_path == None or output_path == None:
    print('Invalid arguments, required arguments are -v (input video) and -o (output video)')
    exit(0)

delete_temp_path()
create_temp_path()

video_in = VideoFileClip(video_path, audio_buffersize=math.inf)
sample_rate = video_in.audio.fps

count = 0

plot.figure()

C = sample_rate / sample_size

# Generate bins for FFT (x-axis)
bin_to_freq = np.vectorize(lambda b: round(b * C, 0))
bins = bin_to_freq(np.array(range(0, sample_size)))

reader = video_in.audio.coreader().reader
reader.seek(0)

audio_in = reader.read_chunk(sample_size)
step_size = 2048
print('Generating frames: 00%', end='')
while audio_in.size > 0:
    percentage_remaining = str(int(step_size * count * 100 / reader.nframes))
    print('\b\b\b', end='')
    t = len(percentage_remaining)
    if len(percentage_remaining) == 2:
        print('%s%%' % percentage_remaining, end='')
    else:
        print('0%s%%' % percentage_remaining, end='')
    # Read sample_size samples of audio and get the left channel
    left_channel = audio_in[:, 0]

    # Get the magnitude of the frequency
    freqs = np.abs(np.fft.fft(left_channel))
    freqs_norm = np.divide(freqs, freqs.max())
    temp_bins = bins
    if left_channel.size < sample_size:
        temp_bins = bin_to_freq(np.array(range(0, left_channel.size)))

    # Plot frequency on x axis and magnitude on y axis
    plot.clf()
    plot.xlim([0,  8000])
    plot.ylim([0, 1.2])
    plot.plot(temp_bins, freqs_norm)
    plot.savefig(os.path.join(temp_path, 'img' + str(count) + '.png'))

    # Increment counters
    count += 1
    reader.seek(step_size * count)
    audio_in = reader.read_chunk(sample_size)

img_clips = []
clip_duration = step_size / sample_rate

for filename in glob.glob(os.path.join(temp_path, '*.png')):
    img_clip = ImageClip(filename, duration=clip_duration)
    img_clips.append(img_clip)

delete_temp_path()

video_clips = concatenate_videoclips(img_clips, method='compose')
video_out = video_clips.set_audio(video_in.audio)
video_out.write_videofile(output_path, fps=60)
