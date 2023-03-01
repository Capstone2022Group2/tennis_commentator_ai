import cv2
import csv
import getopt
import sys
import torch
import numpy as np
import os

# python bin/annotate_video.py -v  no_commit/short_test.mp4 -o short_annotations -f 0


def draw_box(image, data, bgr):
  x_shape, y_shape = image.shape[1], image.shape[0]
  x1, y1, x2, y2 = int(data[0]*x_shape), int(data[1]*y_shape), int(data[2]*x_shape), int(data[3]*y_shape)
  cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 1)
  return image

obj_det_model = torch.hub.load('ultralytics/yolov5', 'custom', 'ai_models/object_detection/trained/object_detect5.pt')  # custom trained model
obj_det_model.conf = 0.4

start_frame = 0
end_frame = 0
video_path = ''
out_path = ''

try:
  opts, args = getopt.getopt(sys.argv[1:],"v:o:f:")
except getopt.GetoptError:
  print('error')
  sys.exit(2)

if '-o' not in [options[0] for options in opts]:
  out_path = 'output'
  

if '-v' not in [options[0] for options in opts]:
  print('Please choose a video to convert')
  sys.exit(2)

for opt, arg in opts:
  if opt == '-o':
    out_path = arg
  elif opt == '-v':
    video_path = arg
  elif opt == '-f':
    start_frame = int(arg)

out_path = f'no_commit/annotations/{out_path}'

if not os.path.exists(out_path):
    os.makedirs(out_path)

annotations = open(f'{out_path}/annotations.csv', 'a', newline='')
writer = csv.writer(annotations)

data_file = open(f'{out_path}/data.txt', 'a')

vidcap = cv2.VideoCapture(video_path)
success,image = vidcap.read()
count = 0
frame = 0

first_frame_data = []
prev_frame_data = []
curr_frame_data = []

while success:
    skip = False
    print('----------------------------------------')
    frame += 1
    if frame < start_frame:
        success,image = vidcap.read()
        continue
    
    det_objects = obj_det_model(image)
    labels, coord = det_objects.xyxyn[0][:, -1].numpy(), det_objects.xyxyn[0][:, :-1].numpy()

    # if no ball detected, don't annotate the frame
    if 0 not in labels:
        print('no ball')
        first_frame_data = prev_frame_data.copy()
        prev_frame_data = []
        success,image = vidcap.read()
        continue
    else:
        # find the ball with the highests confidence value
        i = 0
        highest_conf = 0
        i_value = -1
        for label in labels:
            # only care about checking the ball
            if label == 0:
                if coord[i][4] > highest_conf:
                  highest_conf = coord[i][4]
                  i_value = i
            i += 1
        # only allow annotation if there is ball data for 3 consecutive frames
        if len(prev_frame_data) > 0 and len(first_frame_data) > 0:
            print('ball')
            print(coord[i_value][4])

            # draw where the model thinks the balls are to avoid annotating false positives
            draw_box(image, coord[i_value], (0, 255, 0))
            draw_box(image, prev_frame_data, (0, 255, 255))
            draw_box(image, first_frame_data, (0, 0, 255))
            curr_frame_data = coord[i_value][:-1]
            event_data = np.concatenate([[frame], curr_frame_data, prev_frame_data, first_frame_data])
            event_data = [str(i) + ' ' for i in event_data]
            print(event_data)

            first_frame_data = prev_frame_data.copy()
            prev_frame_data = curr_frame_data.copy()
            
        else:
            # if there is missing data from previous frames, don't annotate and save current data
            print('no previus data')
            first_frame_data = prev_frame_data.copy()
            # remove confidence value from coord array
            prev_frame_data = coord[i_value][:-1]
            skip = True
                   
    # commands on what to do with each frame
    if not skip:
        cv2.imshow('frame', image)
        k = cv2.waitKey(0) # waits until a key is pressed
        if k == ord('h'):
            data_file.writelines(event_data)
            data_file.write('\n')
            writer.writerow(['hit', frame])
        elif k == ord('b'):
            data_file.writelines(event_data)
            data_file.write('\n')
            writer.writerow(['bounce', frame])
        elif k == ord('n'):
            data_file.writelines(event_data)
            data_file.write('\n')
            writer.writerow(['none', frame])
        elif k == ord('s'):
            pass
        elif k == ord('q'):
            end_frame = frame
            break
    
    
        prev_frame_data = curr_frame_data
    success,image = vidcap.read()

annotations.close()
data_file.close()
