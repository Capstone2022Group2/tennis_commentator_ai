import os
import csv
import pandas as pd
import numpy as np


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
                    # print(len(tokens))
                    # if len(tokens) != 14:
                    #     print(tokens)
                    #     continue
                    del tokens[-1]
                    del tokens[0]

                    # if annotations[i] == 'hit':
                    #     tokens = np.concatenate([tokens, [1,0]])
                    # elif annotations[i] == 'bounce':
                    #     tokens = np.concatenate([tokens, [0,1]])
                    # else:
                    #     tokens = np.concatenate([tokens, [0,0]])
                    # if annotations[i] == 'hit':
                    #     tokens = np.concatenate([tokens, [2]])
                    # elif annotations[i] == 'bounce':
                    #     tokens = np.concatenate([tokens, [1]])
                    # else:
                    #     tokens = np.concatenate([tokens, [0]])

                    if annotations[i] == 'hit':
                        tokens = np.concatenate([tokens, [1]])
                    elif annotations[i] == 'bounce':
                        tokens = np.concatenate([tokens, [1]])
                    else:
                        tokens = np.concatenate([tokens, [0]])

                    # tokens = [token for token in tokens]
                    output[i] = tokens
                    #print(tokens)
                i += 1

df = pd.DataFrame.from_dict(output, orient='index')
#df = df.loc[(df!=0).any(axis=1)] # this line deletes all rows that are all zeroes. don't ask me about this. i don't know. but it does work.
with open(os.path.join('no_commit/annotations/3_frame', 'dataset3.txt'), 'w+') as f:   
    f.write(df.to_string(index=False))
                    