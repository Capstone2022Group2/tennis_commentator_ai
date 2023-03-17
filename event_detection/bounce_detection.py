import cv2
#import ai_models.event_detector.event as event_mod
import numpy as np

'''
detects if a bounce occurred using data from prev frame and current frame
returns the current frame's data, and if a bounce was detected
'''
# def detect_bounces(boxes, first_ball_data, prev_ball_data, frame_num, event_det_model):
#   labels, coord = boxes
#   i = 0
#   bounce = False
#   highest_conf = 0
#   i_value = -1

#   if 0 not in labels: 
#     return [], bounce
    
#   for label in labels:
#     # only care about checking the ball
#     if label == 0:
#       if coord[i][4] > highest_conf:
#         highest_conf = coord[i][4]
#         i_value = i
#     i += 1
#   if len(prev_ball_data) > 0 and len(first_ball_data) > 0:
    
#     event_data = np.concatenate([coord[i_value][:-1], prev_ball_data, first_ball_data])
#     results = event_det_model.predict(event_data.reshape(1, -1))
#     print(results)
#     # for neural network
#     # if results[0][1] > results[0][0] and results[0][1] > results[0][2]:
#     #   print('bounce')
#     #   bounce = True

#     # for decision tree
#     if results[0] == 1:
#       print('bounce')
#       bounce = True
#     #   #cv2.imwrite(f'no_commit/event_debug/frame{count}.jpg', image)
#     #   # remove confidence value from coord array
  
#   return coord[i_value][:-1], bounce

# def check_if_ball_inside_player(boxes, ball):
#   labels, coord = boxes
#   players = []
#   i = 0

#   if 0 in labels: 
#     for label in labels:
#       # find all players
#       if label == 3:
#         players.append(coord[i])
#       i += 1
  
#   bx = ball[0]
#   by = ball[1]
#   bw = ball[2] - bx
#   bh = ball[3] - by

#   for player in players:
#     x = player[0]
#     y = player[1]
#     w = player[2] - x
#     h = player[3] - y

#     if x < bx and y < by:
#     # If bottom-right inner box corner is inside the bounding box
#       if bx + bw < x + w and by + bh < y + h:
#           return True
  
#   return False
      

# def get_area(box):
#   x = box[0]
#   y = box[1]
#   w = box[2] - x
#   h = box[3] - y
  
#   return w * h

def get_center(box):
    x = box[0]
    y = box[1]
    w = box[2] - x
    h = box[3] - y

    # center of the ball
    return (int(x)+int(w)//2, int(y)+int(h)//2)

def detect_bounces(boxes, first_ball_data, prev_ball_data, frame_num, event_det_model):
  labels, coord = boxes
  i = 0
  bounce = False
  highest_conf = 0
  i_value = -1

  curr_ball_data = []
  # get current ball data if exists

  if 0 in labels: 
    for label in labels:
      # only care about checking the ball
      if label == 0:
        if coord[i][4] > highest_conf:
          highest_conf = coord[i][4]
          i_value = i
      i += 1
    
    curr_ball_data = coord[i_value][:-1]

  # check for v shape
  if len(prev_ball_data) > 0 and len(first_ball_data) > 0 and len(curr_ball_data) > 0:
    # if middle box center is lower than the others, it is a v shape
    first_center = get_center(first_ball_data)
    prev_center = get_center(prev_ball_data)
    curr_center = get_center(curr_ball_data)

    # TODO maybe add a scaling factor to only detect larger v shapes
    if prev_center[1] > first_center[1] and prev_center[1] > curr_center[1]:
      bounce = True
      print("v-shape detected")
  
  # # check for ball squish
  # if len(prev_ball_data) > 0 and len(first_ball_data) > 0:
  #   # check that the previous ball frame has wider area?

  #   first_area = get_area(first_ball_data)
  #   prev_area = get_area(prev_ball_data)
  #   first_center = get_center(first_ball_data)
  #   prev_center = get_center(prev_ball_data)

  #   if(first_area > prev_area and prev_center > first_center):
  #     bounce = True
  #     print("ball squish detected")

  return curr_ball_data, bounce
