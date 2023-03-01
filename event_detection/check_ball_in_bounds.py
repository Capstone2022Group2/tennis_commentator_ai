# from operator import truediv
import cv2
#import numpy as np
#from sklearn.cluster import KMeans

import torch

from helper_functions.cv_helper import *


def get_side_of_court(det, obj):
    labels, coord = det[:, -1].numpy(), det[:, :-1].numpy()
    n = len(labels)
    
    for i in range(n):
        # court
        if labels[i] == 2:
            net = coord[i]

            net_pos = net[3]
            obj_pos = obj[3]

            # will be + if obj is above net and - if obj is below net
            return net_pos - obj_pos

def checkBall(contours, ball_pos, img):
    b = ball_pos

    x = b[0,0]
    y = b[0,1]
    w = b[0,2] - x
    h = b[0,3] -y

    # center of the ball
    center = (int(x)+int(w)//2, int(y)+int(h)//2)

    for c in contours:
        result = cv2.pointPolygonTest(c, center, False)
        if(result > 0):
            return True
    
    return False
    #cv2.circle(img, center, radius, (255, 255, 0), 2)
    #b = xyxy2xywh(ball_pos)
    #cv2.rectangle(img, (int(b[0,0]), int(b[0,1])), (int(b[0,2]), int(b[0,3])), (36,255,12), 1)

def checkIfBallInBounds(det, imo, hulls, show_data=False):
    #courts = []
    ball_pos = 0

    img = imo.copy()

    #currCropDim = None
    ballInCourt = False
    ball_detected = False
    
    for *xyxy, conf, cls in reversed(det):

        # ball
        if (int(cls) == 0):
            ball_detected = True
            ball_pos = xyxy
            ball_pos = torch.tensor(ball_pos).view(-1, 4)

    if ball_detected:
        for hull in hulls:
            if checkBall(hull, ball_pos, img):
                ballInCourt = True
                break
    
    # Draw court boundry and ball status for debug:
    if(show_data):
        # draw court boundary
        #cv2.drawContours(img,uni_hull,-1,(0,255,0),2)

        # format text
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (450,700)
        fontScale              = 2
        fontColor              = (255,255, 0)
        thickness              = 5
        lineType               = 3

        # draw text for debug
        if ballInCourt:
            cv2.putText(img,'In Court', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
        elif not ball_detected:
            cv2.putText(img,'No Ball', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
        else:
            cv2.putText(img,'Not in Court', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

    
    # # cv2.imshow('final', img)
    # # cv2.waitKey(0) # waits until a key is pressed
    # # cv2.destroyAllWindows() # destroys the window showing image
    
    return img, ballInCourt
   