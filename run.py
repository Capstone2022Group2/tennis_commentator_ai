import math

import torch
import cv2
from time import time
from event_detection.check_ball_in_bounds import *

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from ai_models.audio_detector import audio_utils
from event_detection.bounce_detection import *
from event_detection.court_detection import *
# from event_detection.detect_replay import *
from event_detection.point_detector import PointDetector
from event_detection.commentator import Commentator
from event_detection.replay_detector import ReplayDetector
# from event_detection.check_game_status import *
from helper_functions.cv_helper import plot_boxes
from helper_functions.object_detection import get_objects_with_highest_conf
import ai_models.event_detector.event as event_mod
import tensorflow as tf
import os

obj_det_model = torch.hub.load('ultralytics/yolov5', 'custom', 'ai_models/object_detection/trained/object_detect5.pt')  # custom trained model
obj_det_model.conf = 0.30
#event_det_model = event_mod.load_model('ai_models/event_detector/trained_model')
replay_det_model = event_mod.load_model('ai_models/event_detector/trained_model')

# Images

################TEST#######################################
# url = 'no_commit/event_debug/frame11.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list
# results = obj_det_model (url)
# #det = results.xyxy[0]
# # print(det)
# img = cv2.imread(url)
# boxes = results.xyxy[0][:, -1].numpy(), results.xyxy[0][:, :-1].numpy()

#is_replay(img, boxes)
#labels, coord = results.xyxy[0][:, -1].numpy(), results.xyxy[0][:, :-1].numpy()
#n = len(labels)
# x_shape, y_shape = img.shape[1], img.shape[0]
# for i in range(n):
#   if labels[i] == 0:
#     center = get_center(coord[i])
#     cv2.circle(img, center, 1, (255, 255, 0), 1)
#     # row = coord[i]
#     x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
#     bgr = (0, 255, 0)
#     cv2.line(img, (x1, y2), (x2, y2), bgr, 2)

#image, boundaries = get_court_boundary(det, img, show_data=True)
# my_img = checkIfBallInBounds(det, img)
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
video_path = 'no_commit/test_match.mp4'
player = cv2.VideoCapture(video_path)
success,image = player.read()
height, width, layers = image.shape
size = (width, height)
# ----------------------------------------------------------

out = cv2.VideoWriter('no_commit/full_demo2.avi',cv2.VideoWriter_fourcc(*"MJPG"), 20, size)

video_file = VideoFileClip(video_path)
audio_file = video_file.audio
audio_reader = audio_file.coreader().reader
audio_reader.seek(0)
audio_model = tf.keras.models.load_model('ai_models/audio_detector/trained_model', compile=False)

first_ball_data = []
prev_ball_data = []
curr_ball_data=[]
prev_det = []

count = 0
bounce_count = 0

point_detector = PointDetector()
display_hit = -1
commentator = Commentator()
replay_detector = ReplayDetector()

first_hit = True
display_bounce = -1

while success:
  count +=1
  start_time = time()
  print(f"detecting frame: ${count}")

  # get detected objects
  det_objects = obj_det_model(image)
  boxes = det_objects.xyxy[0][:, -1].numpy(), det_objects.xyxy[0][:, :-1].numpy()
  det = det_objects.xyxy[0]

  highest_conf_objects = get_objects_with_highest_conf(det_objects)
  #print(highest_conf_objects)
  game_status = replay_detector.check_game_status(highest_conf_objects, replay_det_model)
  isreplay = replay_detector.confirm_if_replay(highest_conf_objects, point_detector.points)

  # Allow points to be scored and serves to be detected again 
  if game_status == 0:
    if not point_detector.point_was_scored and not first_hit:
      point_detector.reset()
      point_detector.points += 1
      commentator.set_commentary('point')
    point_detector.point_was_scored = False
    first_hit = True

  # double check for false positive replay
  if game_status == 2:
    replay_detector.replay_frames +=1
    isreplay = replay_detector.confirm_if_replay(highest_conf_objects, point_detector.points)
    if isreplay:
      replay_detector.expected_points += 1
      commentator.set_commentary('replay')
  else:
    replay_detector.replay_frames = 0

  # only do this if the game status is in the middle of a rally
  if game_status == 1:

    # only check for hits until the first hit is detected for now
    if first_hit:
      frame_num = math.floor(count * audio_file.fps / video_file.fps)
      audio_reader.seek(frame_num)
      audio = audio_utils.get_audio(audio_reader)
      hit_detected = audio_utils.predict(audio, audio_model)[0] > 0.5
      if hit_detected:
          commentator.set_commentary('serve')
          first_hit = False
          #display_hit = 0

    # get the status of the current ball data
    #curr_ball_data, bounce_detected = detect_bounces(boxes,first_ball_data, prev_ball_data, count, event_det_model)
    curr_ball_data, bounce_detected = detect_bounces(boxes,first_ball_data, prev_ball_data)

    if len(prev_ball_data) > 0:
      result = get_side_of_court(det, prev_ball_data)
      if result != None:
        point_detector.side_of_court = result

    # check if the ball has switched sides
    if point_detector.side_of_court * point_detector.prev_side_of_court < 0:
      # reset last bounce, since a double bounce can't happen if ball switches sides
      point_detector.side_of_last_bounce = 0

    # draw the last 3 positions of the ball
    if len(curr_ball_data) > 0 and len(prev_ball_data) > 0 and len(first_ball_data) > 0:
      prev = get_center(prev_ball_data)
      first = get_center(first_ball_data)
      curr = get_center(curr_ball_data)

      cv2.circle(image, prev, 3, (255, 255, 0), 1)
      cv2.circle(image, first, 3, (0, 0, 255), 1)
      cv2.circle(image, curr, 3, (0, 255, 0), 1)

    # check if the ball is in bounds
    image, boundaries = get_court_boundary(det, image, show_data=True)
    image, ball_in_bounds = checkIfBallInBounds(prev_det, image, boundaries, show_data=False)

    # keep track of how many frames in a row the ball is out of bounds
    #point_detector.frames_out_of_bounds = point_detector.frames_out_of_bounds + 1 if not ball_in_bounds else 0
    
    # handle bounces
    if bounce_detected and point_detector.buffer < 0: #and not hit_detected:

      # point was scored
      if point_detector.side_of_last_bounce * point_detector.side_of_court > 0 and point_detector.prev_bounce_in_bounds and not point_detector.point_was_scored:
        point_detector.reset()
        point_detector.points += 1
        commentator.set_commentary('point')
        point_detector.point_was_scored = True

      bounce_count += 1
      point_detector.bounce_detected(ball_in_bounds)
      display_bounce = 0
    
    # if the ball has been out of bounds or not detected for a long time, a point was probably scored 
    # if point_detector.frames_out_of_bounds > 110 and not point_detector.point_was_scored and not first_hit:
    #   point_detector.reset()
    #   point_detector.points += 1
    #   commentator.set_commentary('point')
    #   point_detector.point_was_scored = True

    
    
    # if (display_hit <= 5 and display_hit >=0):
    #     cv2.putText(image, "HIT", (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5, 3)
    #     display_hit += 1
    # else:
    #     display_hit = -1

    
    if display_bounce <= 5 and display_bounce >=0:
      
      cv2.putText(image,'bounce', 
              (280,100), 
              cv2.FONT_HERSHEY_SIMPLEX, 
              2,
              (255,255, 0),
              5,
              3)
      
      display_bounce += 1
    else:
      display_bounce = -1

    ### DEBUG ###
    cv2.putText(image,f'points: {point_detector.points}', 
            (50,150), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            (0,0, 255),
            4,
            3)

    point_detector.update()

    first_ball_data = prev_ball_data.copy()
    prev_ball_data = curr_ball_data.copy()
    prev_det = det_objects.xyxy[0]
  
  # display commentary if it exists
  commentator.display_commentary(image)

  game_status_text = 'game' if game_status == 1 else 'replay' if game_status == 2 else 'none'
  print(game_status_text)
  cv2.putText(image,game_status_text, 
            (50,250), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            (255,255, 0),
            5,
            3)
  
  cv2.putText(image,str(count), 
              (50,680), 
              cv2.FONT_HERSHEY_SIMPLEX, 
              2,
              (255,255, 0),
              3,
              3)
  # add frames to the output video
  out.write(image)
  success,image = player.read()

out.release()
