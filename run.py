import torch
import cv2
import pafy
from time import time
from bin.point_detection import *

model = torch.hub.load('ultralytics/yolov5', 'custom', 'trained_models/trained/object_detect5.pt')  # custom trained model

# Images

################TEST#######################################
# url = 'images111/frame1016.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list
# results = model(url)
# det = results.xyxy[0]
# img = cv2.imread(url) 
# my_img = checkIfBallInBounds(det, img)
# cv2.imshow('rect', my_img) 
# cv2.waitKey(0) # waits until a key is pressed


###REAL############

def plot_boxes(results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, model.names[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

#----Detect using URL----
play = pafy.new('https://www.youtube.com/watch?v=oyxhHkOel2I&t=1s').streams[-1]
player = cv2.VideoCapture(play.url)
width = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))

success,image = player.read()

#----Detect using downloaded video----
# player = cv2.VideoCapture('no_commit/short_test.mp4')
# success,image = player.read()
#height, width, layers = image.shape

size = (width, height)

out = cv2.VideoWriter('no_commit/results.avi',cv2.VideoWriter_fourcc(*"MJPG"), 20, size)

count = 0
while success:
  start_time = time()
  print(f"detecting frame: ${count}")

  results = model(image)

  # detect if ball is in bounds and draw court boundary
  det = results.xyxy[0]
  court_boundary = checkIfBallInBounds(det, image)

  # plot object detection boxes
  boxes = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
  objects = plot_boxes(boxes, image)
  
  end_time = time()
  fps = 1/np.round(end_time - start_time, 3)
  print(f"Frames Per Second : {fps}")

  # add frames to the output video
  out.write(court_boundary)
  #out.write(objects)
  count +=1
  
  success,image = player.read()

out.release()
