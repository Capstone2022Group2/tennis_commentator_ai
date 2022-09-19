# *************************************************************************
# You must install the correct version of opencv for VideoCapture to work *
# with pip: pip install opencv-contrib-python                             *
# *************************************************************************

import cv2
import sys
import getopt
import os

frame_to_capture = 1
target_dir = 'data'

try:
  opts, args = getopt.getopt(sys.argv[1:],"v:f:d:")
except getopt.GetoptError:
  print('error')
  sys.exit(2)

if '-v' not in [options[0] for options in opts]:
  print('Please choose a video to convert')
  sys.exit(2)

for opt, arg in opts:
  if opt == '-v':
    video_title = arg
  elif opt == '-f':
    frame_to_capture = int(arg)
  elif opt == '-d':
    target_dir = arg


if not os.path.exists(target_dir):
    os.makedirs(target_dir)


vidcap = cv2.VideoCapture(video_title)
success,image = vidcap.read()
count = 0
img = 0

print('gathering images...')

while success:
  
  if(count % frame_to_capture == 0):
    img += 1
    cv2.imwrite(f'${target_dir}/frame%d.jpg' % img, image)     # save frame as JPEG file 
  
  success,image = vidcap.read()
  count = (count + 1) % frame_to_capture

print(f'created {img} images from {video_title}')