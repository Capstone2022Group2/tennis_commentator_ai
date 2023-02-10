import cv2
import ai_models.event_detector.event as event_mod
import numpy as np

'''
detects if a bounce occurred using data from prev frame and current frame
returns the current frame's data, and if a bounce was detected
'''
def detect_bounces(boxes, prev_frame_data, frame_num, event_det_model):
  labels, coord = boxes
  i = 0
  bounce = False
  for label in labels:
    # only care about checking the ball
    if label == 0:
      if len(prev_frame_data) > 0:
       
        event_data = np.concatenate([[frame_num], coord[i][:-1], prev_frame_data])
        results = event_det_model.predict(event_data.reshape(1, -1))
        print(results)
        if results[0] == 1:
                print('bounce')
                bounce = True
                #cv2.imwrite(f'no_commit/event_debug/frame{count}.jpg', image)
    
      # remove confidence value from coord array
      return coord[i][:-1], bounce
      
    i += 1

  return [], bounce