import os
import types
import pandas as pd
import csv

class DataObj:
    def __init__(self, xcenter, ycenter, width, height, conf):
        self.xcenter = xcenter
        self.ycenter = ycenter
        self.width = width
        self.height = height
        self.conf = conf
        
    def toFrameRow(self):
        return [self.xcenter, self.ycenter, self.width, self.height]
        
emptyBallObj = DataObj(0, 0, 0, 0, 0)
directory = 'labels'
data = {}

# yolov5 prints its output as a series of dataframes with one frame per file, using a standardized naming convention.
# we need to amalgamate that data into a coherent dataframe for the ai.
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        with open(f, "r") as file:
            fileData = file.read()
            lines = fileData.split('\n')
            tokens = filename.split('_')
            framename = tokens[len(tokens) - 1].removesuffix('.txt')
            dataObjs = {}
            
            # dataObjs is a dictionary. each index is a yolov5 class, with ball = 0, court = 1, net = 2, player = 3.
            # each element is a list, since we can detect more than one item of each kind.
            dataObjs[0] = []
            dataObjs[1] = []
            dataObjs[2] = []
            dataObjs[3] = []
            
            for line in lines:
                if line != '':
                    tokens = line.split(' ')
                    
                    # each dataObj is the data corresponding to a singular bounding box
                    # the first element (tokens[0]) is the object class, the rest follow the format x_center, y_center, width, height, confidence
                    dataObj = DataObj(tokens[1], tokens[2], tokens[3], tokens[4], tokens[5])
                    dataObjs[int(tokens[0])].append(dataObj)
            
            data[int(framename)] = dataObjs

annotations = {}

# you need your annotations in a single csv file, with the format "eventType,frameNumber" each on a new line.
with open(os.path.join('dataset', 'annotations.csv'), 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        annotations[int(row[1])] = row[0]
    
outputData = {}
            
for i in range(1, 2545): # should be for i in range(1, data) for a full video annotation. stop it on the frame that you get to when annotating a video.
    if i in data and i-1 in data and i+1 in data and i-2 in data and i+2 in data:
        # we need an empty ball object otherwise python complains.
        if len(data[i][0]) == 0:
            data[i][0].append(emptyBallObj)
        if len(data[i-1][0]) == 0:
            data[i-1][0].append(emptyBallObj)
        if len(data[i-2][0]) == 0:
            data[i-2][0].append(emptyBallObj)
        if len(data[i+1][0]) == 0:
            data[i+1][0].append(emptyBallObj)
        if len(data[i+2][0]) == 0:
            data[i+2][0].append(emptyBallObj)
        # access the first element of the ball (0) index on the ith frame
        dataFrameRow = data[i][0][0].toFrameRow() + data[i-1][0][0].toFrameRow() + data[i-2][0][0].toFrameRow() + data[i+1][0][0].toFrameRow() + data[i+2][0][0].toFrameRow()
        
        # append our annotations to the dataframe. hit is first y column, bounce is second
        if i in annotations:
            if annotations[i] == 'hit':
                dataFrameRow += [1, 0]
            elif annotations[i] == 'bounce':
                dataFrameRow += [0, 1]
            else:
                dataFrameRow += [0, 0]
        else:
            dataFrameRow += [0, 0]
        
        outputData[i] = dataFrameRow
            
# convert to dataframe and output to a file usable by event.py
df = pd.DataFrame.from_dict(outputData, orient='index')
df = df.loc[(df!=0).any(axis=1)] # this line deletes all rows that are all zeroes. don't ask me about this. i don't know. but it does work.
with open(os.path.join('dataset', 'dataset.txt'), 'w+') as f:   
    f.write(df.to_string())