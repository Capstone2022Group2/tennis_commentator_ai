import cv2
#import ai_models.event_detector.event as event_mod
import numpy as np

'''
detects if a bounce occurred using data from prev frame and current frame
returns the current frame's data, and if a bounce was detected
'''
def detect_bounces(boxes, first_ball_data, prev_ball_data, frame_num, event_det_model):
  labels, coord = boxes
  i = 0
  bounce = False
  highest_conf = 0
  i_value = -1

  if 0 not in labels: 
    return [], bounce
    
  for label in labels:
    # only care about checking ball with highest confidence
    if label == 0:
      if coord[i][4] > highest_conf:
        highest_conf = coord[i][4]
        i_value = i
    i += 1

  # only detect bounces if there are 3 consecutive frames of ball data
  if len(prev_ball_data) > 0 and len(first_ball_data) > 0:
    
    event_data = np.concatenate([coord[i_value][:-1], prev_ball_data, first_ball_data])
    results = event_det_model.predict(event_data.reshape(1, -1))
    print(results)
    if results[0] == 1:
      print('bounce')
      bounce = True
      
  # remove confidence value from coord array
  return coord[i_value][:-1], bounce
