import torch
import cv2
import pafy
from time import time
from event_detection.point_detection import *
from event_detection.bounce_detection import *
from event_detection.court_detection import *
from helper_functions.cv_helper import plot_boxes
import ai_models.event_detector.event as event_mod
import os

obj_det_model = torch.hub.load('ultralytics/yolov5', 'custom', 'ai_models/object_detection/trained/object_detect5.pt')  # custom trained model
obj_det_model.conf = 0.40
event_det_model = event_mod.load_model('ai_models/event_detector/trained_model')

# Images

################TEST#######################################
# url = 'no_commit/debug_img.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list
# results = obj_det_model (url)
# det = results.xyxy[0]
# print(det)
# img = cv2.imread(url)
# image, boundaries = get_court_boundary(det, img, show_data=True)
# # my_img = checkIfBallInBounds(det, img)
# #cv2.imshow('rect', my_img) 
# cv2.waitKey(0) # waits until a key is pressed


###REAL############
#----Detect using URL----
# play = pafy.new('https://www.youtube.com/watch?v=oyxhHkOel2I&t=1s').streams[-1]
# player = cv2.VideoCapture(play.url)
# width = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
# success,image = player.read()
# --------------------------------------------------------

#----Detect using downloaded video----
player = cv2.VideoCapture('no_commit/short_test.mp4')
success,image = player.read()
height, width, layers = image.shape
size = (width, height)
# ----------------------------------------------------------

out = cv2.VideoWriter('no_commit/demo_test.avi',cv2.VideoWriter_fourcc(*"MJPG"), 20, size)

first_ball_data = []
prev_ball_data = []
curr_ball_data=[]
prev_det = []

count = 0
display_bounce = -1
in_bounds = False
while success:
  start_time = time()
  print(f"detecting frame: ${count}")

  det_objects = obj_det_model(image)

  # plot object detection boxes
  boxes = det_objects.xyxyn[0][:, -1].numpy(), det_objects.xyxyn[0][:, :-1].numpy()
  #objects_frame = plot_boxes(object_det_model, boxes, image)
  curr_ball_data, bounce_detected = detect_bounces(boxes,first_ball_data, prev_ball_data, count, event_det_model)
  if len(curr_ball_data) > 0:
    draw_box(image, curr_ball_data, (0, 255, 0))
  first_ball_data = prev_ball_data.copy()
  prev_ball_data = curr_ball_data.copy()


  # doing this every frame for debug purpose
  det = det_objects.xyxy[0]
  image, boundaries = get_court_boundary(det, image, show_data=True)
  image, ball_in_bounds = checkIfBallInBounds(prev_det, image, boundaries, show_data=False)

  if bounce_detected:
    display_bounce = 0
    in_bounds = ball_in_bounds
    #detect if ball is in bounds and draw court boundary
    #det = det_objects.xyxy[0]
    #image, ball_in_bounds = checkIfBallInBounds(det, image, show_data=False)
    

  ### DEBUG ###
  if display_bounce <= 5 and display_bounce >=0:
    
    cv2.putText(image,'bounce', 
            (280,100), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            (255,255, 0),
            5,
            3)
    
    text = 'in bounds' if in_bounds else 'not in bounds'
    cv2.putText(image, text, 
            (550,100), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            (255,255, 0),
            5,
            3)
    display_bounce += 1
  else:
    display_bounce = -1


  # add frames to the output video
  out.write(image)
  #out.write(objects_frame)
  count +=1
  prev_det = det
  
  success,image = player.read()

out.release()
