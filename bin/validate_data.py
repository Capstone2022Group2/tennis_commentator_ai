import os
import csv
import pandas as pd
import numpy as np
import cv2
import torch

annotations = {}

def draw_box(image, data, bgr):
  x_shape, y_shape = image.shape[1], image.shape[0]
  x1, y1, x2, y2 = int(data[0]*x_shape), int(data[1]*y_shape), int(data[2]*x_shape), int(data[3]*y_shape)
  cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 1)
  return image

with open(os.path.join('no_commit/annotations/3_frame', 'new_annotations.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        i=0
        for row in reader:
            annotations[i] = [row[0], int(row[1])]
            i+=1

data = {}
with open('no_commit/annotations/3_frame/data.txt', "r") as file:
    fileData = file.read()
    lines = fileData.split('\n')
    
    i = 0
    for line in lines:
        if line != '':
            
            tokens = line.split(' ')
            del tokens[-1]
            del tokens[0]
            data[i] = tokens
            #print(tokens)
        i += 1


obj_det_model = torch.hub.load('ultralytics/yolov5', 'custom', 'ai_models/object_detection/trained/object_detect5.pt')  # custom trained model
obj_det_model.conf = 0.2

# new_annotations = open(f'no_commit/annotations/3_frame/new_annotations2.csv', 'a', newline='')
# writer = csv.writer(new_annotations)

data_file = open(f'no_commit/annotations/3_frame/new_data.txt', 'a')

video_path = 'no_commit/100_shots_small.mp4'
vidcap = cv2.VideoCapture(video_path)
success,image = vidcap.read()
frame_num = 0
index = 975

print("starting")
print(annotations[index][1])
while success:
    frame_num+=1

    if frame_num != int(annotations[index][1]):
        success,image = vidcap.read()
        continue
    
    det_objects = obj_det_model(image)
    labels, coord = det_objects.xyxyn[0][:, -1].numpy(), det_objects.xyxyn[0][:, :-1].numpy()

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
    player1 = coord[i_value][:-1]
    player2 = coord[s_value][:-1]
    if i_value < 0:
        player1 = [0,0,0,0]
    if s_value < 0:
        player2 = [0,0,0,0]
    
    print(player1)
    print(player2)
    player_data = np.concatenate([player1, player2])
                
    draw_box(image, coord[i_value][:-1], (255, 255, 0))
    draw_box(image, coord[s_value][:-1], (255, 255, 0))

    print(player_data)
    curr_frame_data = [float(data[index][0]), float(data[index][1]), float(data[index][2]), float(data[index][3])]
    prev_frame_data = [float(data[index][4]), float(data[index][5]), float(data[index][6]), float(data[index][7])]
    first_frame_data = [float(data[index][8]),float(data[index][9]), float(data[index][10]), float(data[index][11])]
    draw_box(image, first_frame_data, (0, 0, 255))
    draw_box(image, prev_frame_data, (0, 255, 255))
    draw_box(image, curr_frame_data, (0, 255, 0))

    event_data = np.concatenate([[frame_num], curr_frame_data, prev_frame_data, first_frame_data, player_data])
    event_data = [str(i) + ' ' for i in event_data]

    print('-----------------------------------------------------------------')
    print(f'Frame: {frame_num}')
    # print(f'Label: {annotations[index][0]}')

    cv2.imshow('frame', image)
    k = cv2.waitKey(0) # waits until a key is pressed

    # if k == ord('h'):
    #     writer.writerow(['hit', frame_num])
    # elif k == ord('b'):
    #     writer.writerow(['bounce', frame_num])
    # elif k == ord('n'):
    #     writer.writerow(['none',frame_num])
    # elif k == ord('s'):
    #      writer.writerow([annotations[index][0],frame_num])
    # elif k == ord('q'):
    #     end_frame = frame_num
    #     break
    if k == ord('a'):
        data_file.writelines(event_data)
        data_file.write('\n')
    elif k == ord('s'):
        pass
    elif k == ord('q'):
        end_frame = frame_num
        break

    index += 1
    success,image = vidcap.read()

#new_annotations.close()
     