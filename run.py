import math

import torch
import cv2
from time import time
from event_detection.check_ball_in_bounds import *

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from ai_models.audio_detector import audio_utils
from event_detection.point_detection import *
from event_detection.bounce_detection import *
from event_detection.court_detection import *
from event_detection.get_side_of_court import *
from event_detection.point_detector import PointDetector
from helper_functions.cv_helper import plot_boxes
import ai_models.event_detector.event as event_mod
import tensorflow as tf
import os

obj_det_model = torch.hub.load('ultralytics/yolov5', 'custom', 'ai_models/object_detection/trained/object_detect5.pt')  # custom trained model
obj_det_model.conf = 0.20
event_det_model = event_mod.load_model('ai_models/event_detector/trained_model')

# Images

################TEST#######################################
# url = 'no_commit/debug_img.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list
# results = obj_det_model (url)
# #det = results.xyxy[0]
# # print(det)
# img = cv2.imread(url)
# labels, coord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
# n = len(labels)
# x_shape, y_shape = img.shape[1], img.shape[0]
# for i in range(n):
#   if labels[i] == 2:
#     row = coord[i]
#     x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
#     bgr = (0, 255, 0)
#     cv2.line(img, (x1, y2), (x2, y2), bgr, 2)

# #image, boundaries = get_court_boundary(det, img, show_data=True)
# # my_img = checkIfBallInBounds(det, img)
# cv2.imshow('rect', img) 
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
out = cv2.VideoWriter('no_commit/results_fixed.avi',cv2.VideoWriter_fourcc(*"MJPG"), 20, size)
video_file = VideoFileClip('no_commit/short_test.mp4')
audio_file = video_file.audio
audio_reader = audio_file.coreader().reader
audio_reader.seek(0)
audio_model = tf.keras.models.load_model('ai_models/audio_detector/trained_model', compile=False)

count = 0
display_bounce = -1
display_point = -1
bounce_text = ''
bounce_count = 0

point_detector = PointDetector()
display_hit = -1
while success:
  start_time = time()
  print(f"detecting frame: ${count}")

  frame_num = math.floor(count * audio_file.fps / video_file.fps)
  audio = audio_utils.get_audio(audio_reader)
  hit_detected = audio_utils.predict(audio, audio_model)[0] > 0.5
  if hit_detected:
      display_hit = 0

  det_objects = obj_det_model(image)

  # plot object detection boxes
  boxes = det_objects.xyxyn[0][:, -1].numpy(), det_objects.xyxyn[0][:, :-1].numpy()
  #objects_frame = plot_boxes(object_det_model, boxes, image)
  curr_ball_data, bounce_detected = detect_bounces(boxes,first_ball_data, prev_ball_data, count, event_det_model)
  # if len(curr_ball_data) > 0 and len(prev_ball_data) > 0 and len(first_ball_data) > 0:
  #   draw_box(image, curr_ball_data, (0, 255, 0))
  #   draw_box(image, prev_ball_data, (0, 255, 255))
  #   draw_box(image, first_ball_data, (0, 0, 255))
  
  # doing this every frame for debug purpose
  det = det_objects.xyxy[0]
  # image, boundaries = get_court_boundary(det, image, show_data=True)
  # image, ball_in_bounds = checkIfBallInBounds(prev_det, image, boundaries, show_data=False)

  if bounce_detected and point_detector.buffer < 0:
    unscaled_det = det_objects.xyxyn[0]
    
    side_of_court = get_side_of_court(unscaled_det, prev_ball_data)
    if point_detector.prev_bounce_in_bounds:
      #side_of_court = get_side_of_court(unscaled_det, prev_ball_data)
      if point_detector.side_of_court * side_of_court > 0:
        point_detector.reset()
        display_point = 0
        #TODO: figure out who scored the point

    image, boundaries = get_court_boundary(prev_det, image, show_data=False)
    point_detector.prev_bounce_in_bounds = checkIfBallInBounds(prev_det, image, boundaries, show_data=False)
    point_detector.side_of_court = side_of_court


    #TODO: give a point if the current bounce is out of bounds
    
    #   else:
    #     image, boundaries = get_court_boundary(prev_det, image, show_data=False)
    #     point_detector.prev_bounce_in_bounds = checkIfBallInBounds(prev_det, image, boundaries, show_data=False)
    #     point_detector.side_of_court = side_of_court
    # else:
    #   image, boundaries = get_court_boundary(prev_det, image, show_data=False)
    #   point_detector.prev_bounce_in_bounds = checkIfBallInBounds(prev_det, image, boundaries, show_data=False)
    #   point_detector.side_of_court = get_side_of_court(unscaled_det, prev_ball_data)

    # det = det_objects.xyxy[0]
    # image, boundaries = get_court_boundary(det, image, show_data=True)
    # image, ball_in_bounds = checkIfBallInBounds(prev_det, image, boundaries, show_data=False)
    display_bounce = 0
    in_bounds = point_detector.prev_bounce_in_bounds
    bounce_text = 'in bounds' if in_bounds else 'not in bounds'
    bounce_count += 1
    point_detector.buffer = 0
    

  ### DEBUG ###
  if display_bounce <= 5 and display_bounce >=0:
    
    cv2.putText(image,'bounce', 
            (280,100), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            (255,255, 0),
            5,
            3)
    
    #bounce_text = 'in bounds' if in_bounds else 'not in bounds'
    cv2.putText(image, bounce_text, 
            (550,100), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            (255,255, 0),
            5,
            3)
    display_bounce += 1
  else:
    display_bounce = -1

  if display_point <= 5 and display_point >=0:
    
    cv2.putText(image,'point', 
            (280,150), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            (0,0, 255),
            5,
            3)
    
    display_point += 1
  else:
    display_point = -1
  if (display_hit <= 5 and display_hit >=0):
      cv2.putText(image, "HIT", (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5, 3)
      display_hit += 1
  else:
      display_hit = -1


  cv2.putText(image,str(bounce_count), 
            (480,150), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            (0,0, 255),
            5,
            3)
  # add frames to the output video
  out.write(image)
  #out.write(objects_frame)
  count +=1
  prev_det = det

  point_detector.buffer = point_detector.buffer + 1 if point_detector.buffer < 6 else -1
  # point_detector.buffer += 1
  # if point_detector.buffer > 6:
  #   point_detector.buffer = -1

  first_ball_data = prev_ball_data.copy()
  prev_ball_data = curr_ball_data.copy()
  
  success,image = player.read()

out.release()
