import os
import csv
import pandas as pd


annotations = {}
with open(os.path.join('no_commit/annotations/full_annotations', 'annotations.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        i=0
        for row in reader:
            annotations[i] = row[0]
            i+=1

output = {}
with open('no_commit/annotations/full_annotations/data.txt', "r") as file:
            fileData = file.read()
            lines = fileData.split('\n')
            
            i = 0
            for line in lines:
                if line != '':
                    tokens = line.split(' ')
                    del tokens[-1]
                    del tokens[0]

                    if annotations[i] == 'hit':
                        tokens += [1, 0]
                    elif annotations[i] == 'bounce':
                        tokens += [0, 1]
                    else:
                        tokens += [0, 0]

                    tokens = [float(token) for token in tokens]
                    output[i] = tokens
                    print(tokens)
                i += 1

df = pd.DataFrame.from_dict(output, orient='index')
#df = df.loc[(df!=0).any(axis=1)] # this line deletes all rows that are all zeroes. don't ask me about this. i don't know. but it does work.
with open(os.path.join('no_commit/annotations/full_annotations', 'dataset.txt'), 'w+') as f:   
    f.write(df.to_string(index=False))
                    