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
obj_det_model.conf = 0.2

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

annotations = open(f'{out_path}/replay_annotations.csv', 'a', newline='')
writer = csv.writer(annotations)

data_file = open(f'{out_path}/replay_data.txt', 'a')

vidcap = cv2.VideoCapture(video_path)
success,image = vidcap.read()
# count = 0
frame = 0

# first_frame_data = []
# prev_frame_data = []
# curr_frame_data = []


player1 = []
player2 = []
court1 = []
court2 = []
net = []

count = 0
frame_to_capture = 5

while success:
    skip = False
    print('----------------------------------------')
    
    frame += 1
    if frame < start_frame:
        success,image = vidcap.read()
        continue
    
    if(count % frame_to_capture == 0):
        print(f'frame: {frame}')
        det_objects = obj_det_model(image)
        labels, coord = det_objects.xyxyn[0][:, -1].numpy(), det_objects.xyxyn[0][:, :-1].numpy()

        # ball
        i = 0
        highest_conf = 0
        i_value = -1
        ball_data = []
        for label in labels:
            # only care about checking the ball
            if label == 0:
                if coord[i][4] > highest_conf:
                    highest_conf = coord[i][4]
                    i_value = i
            i += 1
        if i_value > -1:
            ball_data = coord[i_value][:-1]
        else:
            ball_data = [0,0,0,0]

        # net
        i = 0
        highest_conf = 0
        i_value = -1
        net = []
        for label in labels:
            # only care about checking the ball
            if label == 2:
                if coord[i][4] > highest_conf:
                    highest_conf = coord[i][4]
                    i_value = i
            i += 1
        if i_value > -1:
            net = coord[i_value][:-1]
        else:
            net = [0,0,0,0]

        # players
        player_data = []
        highest_conf = 0.01
        second_conf = 0
        i_value = -1
        s_value = -2
        i = 0
        for label in labels:
            # only care about checking the ball
            if label == 3:
                if coord[i][4] > highest_conf:
                    second_conf = highest_conf
                    s_value = i_value
                    highest_conf = coord[i][4]
                    i_value = i
                elif coord[i][4] > second_conf:
                    second_conf = coord[i][4]
                    s_value = i
                
            i += 1
        
        player1 = []
        player2 = []
    
        if i_value < 0:
            player1 = [0,0,0,0]
        else:
            player1 = coord[i_value][:-1]
        if s_value < 0:
            player2 = [0,0,0,0]
        else:
            player2 = coord[s_value][:-1]
        
        player_data = np.concatenate([player1, player2])

        # courts
        court_data = []
        highest_conf = 0.01
        second_conf = 0
        i_value = -1
        s_value = -2
        i = 0
        for label in labels:
            # only care about checking the ball
            if label == 1:
                if coord[i][4] > highest_conf:
                    second_conf = highest_conf
                    s_value = i_value
                    highest_conf = coord[i][4]
                    i_value = i
                elif coord[i][4] > second_conf:
                    second_conf = coord[i][4]
                    s_value = i
                
            i += 1
        
        court1 = []
        court2 = []
        
        
        if i_value < 0:
            court1 = [0,0,0,0]
        else:
            court1 = coord[i_value][:-1]
        if s_value < 0:
            court2 = [0,0,0,0]
        else:
            court2 = coord[s_value][:-1]

        court_data = np.concatenate([court1, court2])

        print(f'ball: {ball_data}')
        print(f'player1: {player1}')
        print(f'player2: {player2}')
        print(f'court1: {court1}')
        print(f'court2: {court2}')
        print(f'net: {net}')

        draw_box(image, ball_data, (255, 255, 0))
        draw_box(image, player1, (255, 255, 0))
        draw_box(image, player2, (255, 255, 0))
        draw_box(image, court1, (255, 255, 0))
        draw_box(image, court2, (255, 255, 0))
        draw_box(image, net, (255, 255, 0))

        event_data = np.concatenate([[frame], ball_data, player_data, court_data, net])
        event_data = [str(i) + ' ' for i in event_data]
        print(len(event_data))
        cv2.imshow('frame', image)
        k = cv2.waitKey(0) # waits until a key is pressed
        if k == ord('g'):
            data_file.writelines(event_data)
            data_file.write('\n')
            writer.writerow(['game', frame])
        elif k == ord('r'):
            data_file.writelines(event_data)
            data_file.write('\n')
            writer.writerow(['replay', frame])
        elif k == ord('n'):
            data_file.writelines(event_data)
            data_file.write('\n')
            writer.writerow(['none', frame])
        elif k == ord('s'):
            pass
        elif k == ord('q'):
            end_frame = frame
            break
    
    success,image = vidcap.read()
    count = (count + 1) % frame_to_capture

annotations.close()
data_file.close()
