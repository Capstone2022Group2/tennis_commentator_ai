import math

import torch
import cv2
from time import time
from event_detection.check_ball_in_bounds import *

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from ai_models.audio_detector import audio_utils
#from event_detection.point_detection import *
from event_detection.bounce_detection import *
from event_detection.court_detection import *
#from event_detection.get_side_of_court import *
from event_detection.point_detector import PointDetector
from event_detection.commentator import Commentator
from helper_functions.cv_helper import plot_boxes
import ai_models.event_detector.event as event_mod
import tensorflow as tf
import os

obj_det_model = torch.hub.load('ultralytics/yolov5', 'custom', 'ai_models/object_detection/trained/object_detect5.pt')  # custom trained model
obj_det_model.conf = 0.30
event_det_model = event_mod.load_model('ai_models/event_detector/trained_model')

# Images

################TEST#######################################
# url = 'no_commit/debug_img.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list
# results = obj_det_model (url)
# #det = results.xyxy[0]
# # print(det)
# img = cv2.imread(url)
# labels, coord = results.xyxy[0][:, -1].numpy(), results.xyxy[0][:, :-1].numpy()
# n = len(labels)
# x_shape, y_shape = img.shape[1], img.shape[0]
# for i in range(n):
#   if labels[i] == 0:
#     center = get_center(coord[i])
#     cv2.circle(img, center, 1, (255, 255, 0), 1)
#     # row = coord[i]
    # x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
    # bgr = (0, 255, 0)
    # cv2.line(img, (x1, y2), (x2, y2), bgr, 2)

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
player = cv2.VideoCapture('no_commit/long_match.mp4')
success,image = player.read()
height, width, layers = image.shape
size = (width, height)
# ----------------------------------------------------------

out = cv2.VideoWriter('no_commit/long_match_demo.avi',cv2.VideoWriter_fourcc(*"MJPG"), 20, size)

first_ball_data = []
prev_ball_data = []
curr_ball_data=[]
prev_det = []
#out = cv2.VideoWriter('no_commit/results_fixed.avi',cv2.VideoWriter_fourcc(*"MJPG"), 20, size)
# video_file = VideoFileClip('no_commit/short_test.mp4')
# audio_file = video_file.audio
# audio_reader = audio_file.coreader().reader
# audio_reader.seek(0)
# audio_model = tf.keras.models.load_model('ai_models/audio_detector/trained_model', compile=False)

count = 0
display_bounce = -1
display_point = -1
bounce_text = ''
bounce_count = 0

point_detector = PointDetector()
display_hit = -1
commentator = Commentator()

first_bounce = True
prev_side_of_court = 0
side_of_court =0

while success:
  skip = False
  start_time = time()
  print(f"detecting frame: ${count}")

  # frame_num = math.floor(count * audio_file.fps / video_file.fps)
  # audio = audio_utils.get_audio(audio_reader)
  # hit_detected = audio_utils.predict(audio, audio_model)[0] > 0.5
  # if hit_detected:
  #     display_hit = 0

  det_objects = obj_det_model(image)

  # plot object detection boxes
  #boxes = det_objects.xyxyn[0][:, -1].numpy(), det_objects.xyxyn[0][:, :-1].numpy()
  boxes = det_objects.xyxy[0][:, -1].numpy(), det_objects.xyxy[0][:, :-1].numpy()
  det = det_objects.xyxy[0]

  curr_ball_data, bounce_detected = detect_bounces(boxes,first_ball_data, prev_ball_data, count, event_det_model)

  # if len(curr_ball_data) > 0:
  #   point_detector.frames_no_ball_detected = 0
  # else:
  #   point_detector.frames_no_ball_detected += 1

  if len(prev_ball_data) > 0:
    result = get_side_of_court(det, prev_ball_data)
    if result != None:
      side_of_court = result
  # else:
  #   side_of_court = 0

  
  # check if the ball has switched sides
  if side_of_court * prev_side_of_court < 0:
    #print('side switch')
    #point_detector.side_of_court = side_of_court
    # reset last bounce, since a double bounce can't happen if ball switches sides
    point_detector.side_of_last_bounce = 0

  
  if len(curr_ball_data) > 0 and len(prev_ball_data) > 0 and len(first_ball_data) > 0:
    # print(curr_ball_data)
    # draw_box(image, curr_ball_data, (0, 255, 0))
    # draw_box(image, prev_ball_data, (0, 255, 255))
    # draw_box(image, first_ball_data, (0, 0, 255))
    prev = get_center(prev_ball_data)
    first = get_center(first_ball_data)
    curr = get_center(curr_ball_data)
    #print(prev)
    cv2.circle(image, prev, 3, (255, 255, 0), 1)
    cv2.circle(image, first, 3, (0, 0, 255), 1)
    cv2.circle(image, curr, 3, (0, 255, 0), 1)
  
  # doing this every frame for debug purpose
  
  image, boundaries = get_court_boundary(det, image, show_data=True)
  image, ball_in_bounds = checkIfBallInBounds(prev_det, image, boundaries, show_data=False)

  if ball_in_bounds:
    point_detector.frames_out_of_bounds = 0
  else:
    point_detector.frames_out_of_bounds += 1
  
  if bounce_detected and point_detector.buffer < 0: #and not hit_detected:
    #is_hit = check_if_ball_inside_player(boxes, prev_ball_data)
    #print('bounce')
    # temporary until hit detection is implemented
    # TODO change this to happen on first hit
    if first_bounce:
      display_point = 0
      commentator.get_commentary('serve')
      first_bounce = False
      point_detector.reset()
      skip = True
    
    #side_of_court = get_side_of_court(unscaled_det, prev_ball_data)
    

    if point_detector.prev_bounce_in_bounds:
      #side_of_court = get_side_of_court(unscaled_det, prev_ball_data)
      # print('prev bounce in bouncs')
      # print(point_detector.side_of_last_bounce)
      # print(side_of_court)
      # point was scored
      if point_detector.side_of_last_bounce * side_of_court > 0:
        point_detector.reset()
        point_detector.points += 1
        display_point = 0
        commentator.get_commentary('point')
        point_detector.point_was_scored = True

    elif not skip:
      #image, boundaries = get_court_boundary(prev_det, image, show_data=False)
      point_detector.prev_bounce_in_bounds = ball_in_bounds
      
        
        #TODO: figure out who scored the point


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
    point_detector.side_of_last_bounce = side_of_court
    display_bounce = 0
    in_bounds = point_detector.prev_bounce_in_bounds
    bounce_text = 'in bounds' if in_bounds else 'not in bounds'
    bounce_count += 1
    point_detector.buffer = 0
    #point_detector.side_of_court = side_of_court
  
  # if the ball has been out of bounds or not detected for a long time, a point was probably scored 
  if point_detector.frames_out_of_bounds > 60 and not point_detector.point_was_scored and not first_bounce:
    point_detector.reset()
    point_detector.points += 1
    display_point = 0
    commentator.get_commentary('point')
    point_detector.point_was_scored = True

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

  if display_point <= 20 and display_point >=0:
    commentator.display_commentary(image)
    display_point += 1
  else:
    display_point = -1
  if (display_hit <= 5 and display_hit >=0):
      cv2.putText(image, "HIT", (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5, 3)
      display_hit += 1
  else:
      display_hit = -1

  cv2.putText(image,f'points: {point_detector.points}', 
          (50,150), 
          cv2.FONT_HERSHEY_SIMPLEX, 
          2,
          (0,0, 255),
          4,
          3)

  cv2.putText(image,str(bounce_count), 
            (80,680), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            (0,0, 255),
            3,
            3)
  
  cv2.putText(image,str(count), 
            (120,680), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            (255,255, 0),
            3,
            3)
  # add frames to the output video
  out.write(image)
  #out.write(objects_frame)
  count +=1
  prev_det = det_objects.xyxy[0]

  point_detector.buffer = point_detector.buffer + 1 if point_detector.buffer < 6 and point_detector.buffer >= 0 else -1

  first_ball_data = prev_ball_data.copy()
  prev_ball_data = curr_ball_data.copy()

  prev_side_of_court = side_of_court if side_of_court != None else 0
  
  success,image = player.read()

out.release()
