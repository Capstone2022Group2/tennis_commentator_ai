from operator import truediv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

#import torch

from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box

from collections import defaultdict
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))

    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # Get angles in [0, pi] radians
    angles = np.array([line[0][1] for line in lines])

    # Multiply the angles by two and find coordinates of that angle on the Unit Circle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype=np.float32)

    # Run k-means
        # python 3.x, syntax has changed.
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]

    labels = labels.reshape(-1) # Transpose to row vector

    # Segment lines based on their label of 0 or 1
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)

    segmented = list(segmented.values())
    print("Segmented lines into two groups: %d, %d" % (len(segmented[0]), len(segmented[1])))
    return segmented

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


def checkIfBallInBounds(det, imc):
    print("data########################################################################################################")
    #print(det)
    
    courts = []

    img = imc.copy()

    

    for *xyxy, conf, cls in reversed(det):
        if (int(cls)) == 1:
            cropped_image = save_one_box(xyxy, img, file='result.jpg', BGR=True, save=False)
            #cropped_image = img[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::-1]
            courts.append(cropped_image)
            break

            
            # cv2.waitKey(0) # waits until a key is pressed
            # cv2.destroyAllWindows() # destroys the window showing image
    i=0
    for court in courts:
        cropped_image = court
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
        #start = 0
        # most represented color
        # print(colors[len(colors)-1][1])
        # print(colors[len(colors)-4][1])

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hsv_color1 = cv2.cvtColor(np.uint8([[colors[len(colors)-1][1]]]), cv2.COLOR_RGB2HSV)[0][0]
        hsv_color2 = cv2.cvtColor(np.uint8([[colors[len(colors)-1][1]]]), cv2.COLOR_RGB2HSV)[0][0]

        origional = hsv_color1

        hsv_color2[0] = origional[0] + 10
        hsv_color2[1] = 255
        hsv_color2[2] = 255

        hsv_color1[0] = origional[0] - 10
        hsv_color1[1] = 100
        hsv_color1[2] = 100

        # ORANGE_MIN = np.array([300, 20, 100],np.uint8)
        # ORANGE_MAX = np.array([263,44, 51],np.uint8)

        mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)
        #blur_mask = cv2.GaussianBlur(mask,(5, 5),0)
        #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        #ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [c for c in contours if not eliminateContour(c)]
        #big_contour = max(contours, key=cv2.contourArea)

        for c in contours:
            peri = cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, 0.001 * peri, True)

        # contour_img = np.zeros_like(mask)
        #cv2.drawContours(img, contours, -1, (60, 200, 200), 2)
        
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
        # dilate = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
        # edge = cv2.Canny(dilate, 0, 200)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            
            #cv2.putText(img, str(w), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 1)


        #cv2.drawContours(img, contours, -1, (60, 200, 200), 2)
        cv2.imshow('contour', img)
        #mask2 = cv2.inRange(img_hsv, hsv_color4, hsv_color3)
        #res = cv2.bitwise_or(mask,mask, mask= mask2)
        #cv2.imwrite('output2.jpg', mask)
        
        # plt.imshow(mask, cmap='gray')   # this colormap will display in black / white
        # plt.savefig('mygraph.png')

        # res = cv2.bitwise_and(img_hsv, img, mask=mask)
        # cv2.imshow("mask", res)

        # visualize = visualize_colors(cluster, cluster.cluster_centers_)
        # visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
        #cv2.imshow('visualize', visualize)
    #     #cropped_image = cv2.imread('sudoku.jpg')
    #     #cropped_image = img[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::-1]

    #     #cv2.imshow("cropped", cropped_image)
    #     gray = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)

    #     kernel_size = 5
    #     blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    #     # gray = np.float32(gray)
    #     # dst = cv2.cornerHarris(gray,2,3,0.04)
    #     # #result is dilated for marking the corners, not important
    #     # dst = cv2.dilate(dst,None)
    #     # # Threshold for an optimal value, it may vary depending on the image.
    #     # cropped_image[dst>0.01*dst.max()]=[0,0,255]
    #     # cv2.imshow('dst',cropped_image)

    #     # ---For subpix -----
    #     # ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
    #     # dst = np.uint8(dst)
    #     # # find centroids
    #     # ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    #     # # define the criteria to stop and refine the corners
    #     # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    #     # corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    #     low_threshold = 200
    #     high_threshold = 250
    #     edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    #     rho = 1  # distance resolution in pixels of the Hough grid
    #     theta = np.pi / 180  # angular resolution in radians of the Hough grid
    #     threshold = 100  # minimum number of votes (intersections in Hough grid cell)
    #     min_line_length = 50  # minimum number of pixels making up a line
    #     max_line_gap = 20  # maximum gap in pixels between connectable line segments
    #     line_image = np.copy(cropped_image) * 0  # creating a blank to draw lines on

    #     # Run Hough on edge detected image
    #     # Output "lines" is an array containing endpoints of detected line segments
    #     # lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
    #     #                     min_line_length, max_line_gap)
    #     lines = cv2.HoughLines(edges, rho, theta, threshold)

    #     segmented = segment_by_angle_kmeans(lines)
    #     print(segmented[0])
    #     print("----")
    #     print(segmented[1])

        
    #     for line in segmented[0]:
    #         # for x1,y1,x2,y2 in line:
    #         #     cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)
            
    #         for rho,theta in line:
    #             a = np.cos(theta)
    #             b = np.sin(theta)
    #             x0 = a*rho
    #             y0 = b*rho
    #             x1 = int(x0 + 1000*(-b))
    #             y1 = int(y0 + 1000*(a))
    #             x2 = int(x0 - 1000*(-b))
    #             y2 = int(y0 - 1000*(a))
    #             cv2.line(cropped_image, (x1,y1), (x2,y2), (255,0,0), 1)
        
    #     for line in segmented[1]:
    #         # for x1,y1,x2,y2 in line:
    #         #     cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)
    #         for rho,theta in line:
    #             a = np.cos(theta)
    #             b = np.sin(theta)
    #             x0 = a*rho
    #             y0 = b*rho
    #             x1 = int(x0 + 1000*(-b))
    #             y1 = int(y0 + 1000*(a))
    #             x2 = int(x0 - 1000*(-b))
    #             y2 = int(y0 - 1000*(a))
    #             cv2.line(cropped_image, (x1,y1), (x2,y2), (0,255,0), 1)

    #     # for line1 in segmented[1]:
    #     #     for x1,y1,x2,y2 in line1:
    #     #         cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),3)
        
    #     # Draw the lines on the  image
    #     #lines_edges = cv2.addWeighted(cropped_image, 0.8, line_image, 1, 0)
        
    #     i+=1

    #     cv2.imshow("lines"+str(i), cropped_image)

    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image


   