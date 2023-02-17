import cv2
import numpy as np
from sklearn.cluster import KMeans

import torch

from helper_functions.cv_helper import *

# Eliminates contours that are smaller than an arbitrary size.  Needs fine tuning
def eliminateContour(c):
    x,y,w,h = cv2.boundingRect(c)
    if(w < 200 or h < 30):
        return True
    return False

# Takes a RGB color and converts to HSV.  Then modifies that value to get and upper and lower bound of acceptable colors
# Values are derived from trial and error for best results
def getHSVColorRange(color):
    hsv_color1 = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    hsv_color2 = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]

    origional1 = hsv_color1.copy()
    origional2 = hsv_color2.copy()

    # highest value of color we want to accept
    hsv_color2[0] = (origional2[0] + 7) if origional2[0] + 7 <= 180 else 180 # H value cannot exceed 180
    hsv_color2[1] = (origional2[1] + 30) if origional2[1] + 30 <= 255 else 255 # S value cannot exceed 255
    hsv_color2[2] = (origional2[2] + 40) if origional2[2] + 40 <= 255 else 255 

    # lowest value
    hsv_color1[0] = (origional1[0] - 7) if origional1[0] - 7 >= 0 else 0
    hsv_color1[1] = (origional1[1] - 30) if origional1[1] - 30 >= 15 else 15 # S value arbitrarily set to min of 15
    hsv_color1[2] = 35 # V value arbitrarily set to 30 to avoid black colors being allowed

    # ----- visualize colors for debug ------

    # rect = np.zeros((50, 300, 3), dtype=np.uint8)
    # cv2.rectangle(rect, (0, 0), (50, 50), \
    #                   hsv_color1.astype("uint8").tolist(), -1)
    # cv2.rectangle(rect, (50, 0), (100, 50), \
    #                 hsv_color2.astype("uint8").tolist(), -1)
    # visualize = cv2.cvtColor(rect, cv2.COLOR_HSV2RGB)
    # cv2.imshow('hsv', visualize)

    return hsv_color1, hsv_color2  

def get_court_boundary(det, imo, show_data=False):
    courts = []
    uni_hulls = []
    img = imo.copy()

    currCropDim = None
    for *xyxy, conf, cls in reversed(det):
        
        # court
        if (int(cls)) == 1:
            # save the cropped image's dimensions so we can scale contours to full size later
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
        #cv2.imshow('res mask', resMask)

        # arbitrarily choosing small rectangle structuring element to widen the gap between lines of the court
        # This makes is easier to eliminate the doubles area from the contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))
        resMask = cv2.morphologyEx(resMask, cv2.MORPH_OPEN, kernel)

        # get the largest contours of the mask.  This should be the outline of the court (more or less)
        contours = cv2.findContours(resMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [c for c in contours if not eliminateContour(c)]

        # need to patch up any space left between the contours like the net, players and lines
        # Choose a large structuring element because irrelevent contours should be eliminated by now
        bmask = np.zeros((cropped_image.shape[0], cropped_image.shape[1]), np.uint8)
        bmask = cv2.drawContours(bmask,contours,-1,255, -1)
        #cv2.imshow('b mask', bmask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 22))
        dilate_mask = cv2.morphologyEx(bmask, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow('dialate mask', dilate_mask)
        #cv2.waitKey(0) # waits until a key is pressed

        court_boundry = cv2.findContours(dilate_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

        # scale the contours so that the poisiton is relative to the origional image instead of the cropped image
        # Do this by offsetting each point in the contour by the location of the cropped image's position relative to the full img
        for c in court_boundry:
            for point in c:
                for point2 in point:
                    point2[1] += int(cropDim[0, 1])
                    point2[0] += int(cropDim[0, 0])

        # create a convex hull around contours to get a smooth court boundary
        length = len(court_boundry)
        # concatinate points from all shapes into one array
        
        if(length > 0):
            cont = np.vstack([court_boundry[i] for i in range(length)])
            hull = [cv2.convexHull(cont)] # <- array as first element of list
            uni_hulls.append(hull) 
            if(show_data):
                # draw court boundary
                cv2.drawContours(img,hull,-1,(0,255,0),2)
            #cv2.imshow('image', img)
        cn +=1

   
    # cv2.imshow('final', img)
    #cv2.waitKey(0) # waits until a key is pressed
    # cv2.destroyAllWindows() # destroys the window showing image
    
    return img, uni_hulls