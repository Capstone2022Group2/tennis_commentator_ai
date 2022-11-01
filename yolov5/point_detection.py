from operator import truediv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch

from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh,xywh2xyxy, clip_boxes)
from utils.plots import Annotator, colors, save_one_box

from collections import defaultdict

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
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect

def eliminateContour(c):
    x,y,w,h = cv2.boundingRect(c)
    if(w < 200):
        return True
    return False

def getHSVColorRange(color):
    hsv_color1 = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    hsv_color2 = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]

    origional = hsv_color1

    hsv_color2[0] = origional[0] + 10
    hsv_color2[1] = 255
    hsv_color2[2] = 255

    hsv_color1[0] = origional[0] - 10
    hsv_color1[1] = 100
    hsv_color1[2] = 100

    return hsv_color1, hsv_color2

def checkBall(contours, ball_pos, img):
    print(ball_pos)
    b = ball_pos

    x = b[0,0]
    y = b[0,1]
    w = b[0,2] - x
    h = b[0,3] -y
    # below circle to denote mid point of center line
    center = (int(x)+int(w)//2, int(y)+int(h)//2)
    #radius = 2

    for c in contours:
        result = cv2.pointPolygonTest(c, center, False)
        if(result > 0):
            print("ball detected in court")
    #cv2.circle(img, center, radius, (255, 255, 0), 2)
    #b = xyxy2xywh(ball_pos)
    #cv2.rectangle(img, (int(b[0,0]), int(b[0,1])), (int(b[0,2]), int(b[0,3])), (36,255,12), 1)



def checkIfBallInBounds(det, imc):
    print("data########################################################################################################")
    #print(det)
    
    courts = []
    ball_pos = 0

    img = imc.copy()

    currCropDim = None
    

    for *xyxy, conf, cls in reversed(det):
        # court
        if (int(cls)) == 1:
            # save the cropped image's position so we can place it onto the origional image later
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
            ball_pos = xyxy
            ball_pos = torch.tensor(ball_pos).view(-1, 4)

            
            # cv2.waitKey(0) # waits until a key is pressed
            # cv2.destroyAllWindows() # destroys the window showing image

    for (court, cropDim) in courts:
        cropped_image = court
        contourImg = img.copy()
        #img2 = cropped_image.copy()
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        reshape = cropped_image.reshape((cropped_image.shape[0] * cropped_image.shape[1], 3))

        # Find and display most dominant colors
        cluster = KMeans(n_clusters=5).fit(reshape)

        labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        (hist, _) = np.histogram(cluster.labels_, bins = labels)
        hist = hist.astype("float")
        hist /= hist.sum()

        colors = sorted([(percent, color) for (percent, color) in zip(hist, cluster.cluster_centers_)])

        #img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)

        hsv_color1, hsv_color2 = getHSVColorRange(colors[len(colors)-1][1])
        hsv_color3, hsv_color4 = getHSVColorRange(colors[len(colors)-2][1])
        # hsv_color5, hsv_color6 = getHSVColorRange(colors[len(colors)-3][1])
        # hsv_color7, hsv_color8 = getHSVColorRange(colors[len(colors)-4][1])
        # hsv_color9, hsv_color10 = getHSVColorRange(colors[len(colors)-5][1])


        mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)
        mask2 = cv2.inRange(img_hsv, hsv_color3, hsv_color4)
        # mask3 = cv2.inRange(img_hsv, hsv_color5, hsv_color6)
        # mask4 = cv2.inRange(img_hsv, hsv_color7, hsv_color8)
        # mask5 = cv2.inRange(img_hsv, hsv_color9, hsv_color10)
        resMask = mask | mask2 
        #blur_mask = cv2.GaussianBlur(mask,(5, 5),0)
        #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        #ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours = cv2.findContours(resMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [c for c in contours if not eliminateContour(c)]
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
        for c in contours:
            for point in c:
                for point2 in point:
                    point2[1] += int(cropDim[0, 1])
                    point2[0] += int(cropDim[0, 0])
                   

            x,y,w,h = cv2.boundingRect(c)
            #cv2.putText(img, str(w), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (36,255,12), 1)

        checkBall(contours, ball_pos, img)

        #cv2.drawContours(cropped_image, contours, -1, (60, 200, 200), 2)
        cv2.drawContours(img, contours, -1, (60, 200, 200), 2)




        # place the cropped image back into the origional image
        #img[int(cropDim[0, 1]):int(cropDim[0, 3]), int(cropDim[0, 0]):int(cropDim[0, 2]), ::-1] = cropped_image
        #cv2.imshow('contour', contourImg)
        cv2.imshow('box', img)
        #cv2.imshow('mask', resMask)
        #cv2.imshow('box', edge)

        # plt.imshow(mask, cmap='gray')   # this colormap will display in black / white
        # plt.savefig('mygraph.png')

        # res = cv2.bitwise_and(img_hsv, img, mask=mask)
        # cv2.imshow("mask", res)

        # visualize = visualize_colors(cluster, cluster.cluster_centers_)
        # visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
        #cv2.imshow('visualize', visualize)
    

    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image


   