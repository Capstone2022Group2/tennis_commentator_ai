import cv2
import numpy as np
import torch

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

            # cv2.imshow("cropped", cropped_image)
            # cv2.waitKey(0) # waits until a key is pressed
            # cv2.destroyAllWindows() # destroys the window showing image
    i=0
    for court in courts:
        cropped_image = court
        #cropped_image = cv2.imread('sudoku.jpg')
        #cropped_image = img[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::-1]

        #cv2.imshow("cropped", cropped_image)
        gray = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)

        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

        # gray = np.float32(gray)
        # dst = cv2.cornerHarris(gray,2,3,0.04)
        # #result is dilated for marking the corners, not important
        # dst = cv2.dilate(dst,None)
        # # Threshold for an optimal value, it may vary depending on the image.
        # cropped_image[dst>0.01*dst.max()]=[0,0,255]
        # cv2.imshow('dst',cropped_image)

        low_threshold = 200
        high_threshold = 250
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 100  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(cropped_image) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        # lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
        #                     min_line_length, max_line_gap)
        lines = cv2.HoughLines(edges, rho, theta, threshold)

        segmented = segment_by_angle_kmeans(lines)
        print(segmented[0])
        print("----")
        print(segmented[1])

        
        for line in segmented[0]:
            # for x1,y1,x2,y2 in line:
            #     cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)
            
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(cropped_image, (x1,y1), (x2,y2), (255,0,0), 1)
        
        for line in segmented[1]:
            # for x1,y1,x2,y2 in line:
            #     cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(cropped_image, (x1,y1), (x2,y2), (0,255,0), 1)

        # for line1 in segmented[1]:
        #     for x1,y1,x2,y2 in line1:
        #         cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),3)
        
        # Draw the lines on the  image
        #lines_edges = cv2.addWeighted(cropped_image, 0.8, line_image, 1, 0)
        
        i+=1

        cv2.imshow("lines"+str(i), cropped_image)

    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image


   