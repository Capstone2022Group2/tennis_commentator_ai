from operator import truediv
import cv2
import numpy as np
from sklearn.cluster import KMeans

import torch

from utils.general import (cv2, xyxy2xywh,xywh2xyxy, clip_boxes)
from utils.plots import save_one_box

#from collections import defaultdict

def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    for (percent, color) in colors:
        #print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect

# Eliminates contours that are smaller than an arbitrary size.  Needs fine tuning
def eliminateContour(c):
    x,y,w,h = cv2.boundingRect(c)
    if(w < 200):
        return True
    return False

# Takes a RGB color and converts to HSV.  Then modifies that value to get and upper and lower bound of acceptable colors
# Values are derived from trial and error for best results
def getHSVColorRange(color):
    # convert color to HSV
    hsv_color1 = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    hsv_color2 = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]

    print(hsv_color1, hsv_color2)

    origional = hsv_color1.copy()

    # highest value of color we want to accept
    hsv_color2[0] = (origional[0] + 7) if origional[0] + 7 <= 180 else 180 # H value cannot exceed 180
    hsv_color2[1] = (origional[1] + 40) if origional[1] + 40 <= 255 else 255 # S value cannot exceed 255
    hsv_color2[2] = 255

    hsv_color1[0] = (origional[0] - 7) if origional[0] - 7 >= 0 else 0
    hsv_color1[1] = (origional[1] - 40) if origional[1] - 40 >= 15 else 15 # S value arbitrarily set to min of 15
    hsv_color1[2] = 30 # V value arbitrarily set to 30 to avoid black colors being allowed

    #print(hsv_color1, hsv_color2)

    # ----- visualize colors for debug ------

    # rect = np.zeros((50, 300, 3), dtype=np.uint8)
    # cv2.rectangle(rect, (0, 0), (50, 50), \
    #                   hsv_color1.astype("uint8").tolist(), -1)
    # cv2.rectangle(rect, (50, 0), (100, 50), \
    #                 hsv_color2.astype("uint8").tolist(), -1)
    # visualize = cv2.cvtColor(rect, cv2.COLOR_HSV2RGB)
    # cv2.imshow('hsv', visualize)

    return hsv_color1, hsv_color2

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



def checkIfBallInBounds(det, imc):
    courts = []
    ball_pos = 0

    img = imc.copy()

    currCropDim = None
    ballInCourt = False
    ball_detected = False
    

    for *xyxy, conf, cls in reversed(det):
        # court
        if (int(cls)) == 1:
            # save the cropped image's position so we can place it onto the origional image later
            # below code taken from save_one_box in yolov5/utils/plots.py
            currCropDim = xyxy
            currCropDim = torch.tensor(currCropDim).view(-1, 4)
            b = xyxy2xywh(currCropDim)  # boxes
            b[:, 2:] = b[:, 2:] * 1.02 + 10  # box wh * gain + pad
            currCropDim = xywh2xyxy(b).long()
            clip_boxes(currCropDim, img.shape)

            cropped_image = save_one_box(xyxy, img, file='result.jpg', BGR=True, save=False)

            courtImg = [cropped_image, currCropDim]
            courts.append(courtImg)

        # ball
        if (int(cls) == 0):
            ball_detected = True
            ball_pos = xyxy
            ball_pos = torch.tensor(ball_pos).view(-1, 4)

    cn = 0 # keep track of which court we are working on
    for (court, cropDim) in courts:
        cropped_image = court
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        reshape = cropped_image.reshape((cropped_image.shape[0] * cropped_image.shape[1], 3))

        # Find and display most dominant colors in the cropped image using kmeans cluster
        cluster = KMeans(n_clusters=5).fit(reshape)

        labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        (hist, _) = np.histogram(cluster.labels_, bins = labels)
        hist = hist.astype("float")
        hist /= hist.sum()

        colors = sorted([(percent, color) for (percent, color) in zip(hist, cluster.cluster_centers_)])
        img_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)

        # get range of hsv colors for the two most dominant colors in the cropped image
        hsv_color1, hsv_color2 = getHSVColorRange(colors[len(colors)-1][1])
        hsv_color3, hsv_color4 = getHSVColorRange(colors[len(colors)-2][1])

        # create a black and white mask that only highlights the part of the image within the color ranges
        mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)
        mask2 = cv2.inRange(img_hsv, hsv_color3, hsv_color4)
        resMask = mask | mask2
        #cv2.imshow('mask'+str(cn), resMask)
        #blur_mask = cv2.GaussianBlur(mask,(5, 5),0)

        # get the largest contours of the mask.  This should be the outline of the court (more or less)
        contours = cv2.findContours(resMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [c for c in contours if not eliminateContour(c)]

        # TODO make contour smoothing work
        #big_contour = max(contours, key=cv2.contourArea)

        # for c in contours:
        #     peri = cv2.arcLength(c, True)
        #     c = cv2.approxPolyDP(c, 0.001 * peri, True)

        #contour_img = np.zeros_like(mask)
        #cv2.drawContours(cropped_image, contours, -1, (60, 200, 200), 2)
        
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
        # dilate = cv2.morphologyEx(cropped_image, cv2.MORPH_DILATE, kernel)
        # edge = cv2.Canny(dilate, 0, 200)

        # scale the contours so that the poisiton is relative to the origional image instead of the cropped image
        # Do this by offsetting each point in the contour by the location of the cropped image's position
        i = 0
        for c in contours:
            for point in c:
                for point2 in point:
                    point2[1] += int(cropDim[0, 1])
                    point2[0] += int(cropDim[0, 0])
                   

            # x,y,w,h = cv2.boundingRect(c)
            # cv2.putText(img, str(w), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            # cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (36,255,12), 1)
            # cv2.imshow('boxes'+ str(cn) + str(i), cropped_image)
            # i+=1

        if ball_detected:
            if checkBall(contours, ball_pos, img):
                ballInCourt = True


        #cv2.drawContours(cropped_image, contours, -1, (60, 200, 200), 2)
        cv2.drawContours(img, contours, -1, (60, 200, 200), 1)




        # place the cropped image back into the origional image
        #img[int(cropDim[0, 1]):int(cropDim[0, 3]), int(cropDim[0, 0]):int(cropDim[0, 2]), ::-1] = cropped_image

        #cv2.imshow('contour', contourImg)
        cv2.imshow('box', img)
        
        # --- Debugging hsv conversion ---
        cn +=1
        #cv2.imshow('contour', contourImg)
       
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (450,700)
    fontScale              = 2
    fontColor              = (255,255, 0)
    thickness              = 5
    lineType               = 3
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

    
    # cv2.imshow('final', img)
    # cv2.waitKey(0) # waits until a key is pressed
    # cv2.destroyAllWindows() # destroys the window showing image
    
    return img
   