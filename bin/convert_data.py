import os
import csv
import pandas as pd
import numpy as np

'''
creates a csv file with 3 frames of ball data

Each frame is four columns x1 y1 x2 y2, 12 coloumns total

Adds the annotations from annotations.csv to the dataset

'''

annotations = {}
with open(os.path.join('no_commit/annotations/3_frame', 'annotations.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        i=0
        for row in reader:
            annotations[i] = row[0]
            i+=1

output = {}
with open('no_commit/annotations/3_frame/data.txt', "r") as file:
            fileData = file.read()
            lines = fileData.split('\n')
            
            i = 0
            for line in lines:
                if line != '':
                    
                    tokens = line.split(' ')
                    
                    # the annotator script adds a space to each row so we just remove it here
                    del tokens[-1]
                    # remove the frame number from the dataset.  I don't think the model needs this
                    del tokens[0]
                    

                    # Add the labels to the dataset
                    # Just experimenting with different labels here
                    if annotations[i] == 'hit':
                        tokens = np.concatenate([tokens, [1,0]])
                    elif annotations[i] == 'bounce':
                        tokens = np.concatenate([tokens, [0,1]])
                    else:
                        tokens = np.concatenate([tokens, [0,0]])
                    # if annotations[i] == 'hit':
                    #     tokens = np.concatenate([tokens, [2]])
                    # elif annotations[i] == 'bounce':
                    #     tokens = np.concatenate([tokens, [1]])
                    # else:
                    #     tokens = np.concatenate([tokens, [0]])

                    # if annotations[i] == 'hit':
                    #     tokens = np.concatenate([tokens, [1]])
                    # elif annotations[i] == 'bounce':
                    #     tokens = np.concatenate([tokens, [1]])
                    # else:
                    #     tokens = np.concatenate([tokens, [0]])

                    # tokens = [token for token in tokens]
                    output[i] = tokens
                    #print(tokens)
                i += 1

# output to pd dataframe
df = pd.DataFrame.from_dict(output, orient='index')
with open(os.path.join('no_commit/annotations/3_frame', 'dataset3.txt'), 'w+') as f:   
    f.write(df.to_string(index=False))
                    