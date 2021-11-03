'''
Purpose: Generate dataset for depth estimation
'''
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

class_translation = {'Car': 3, 'Van': 4, \
    'Truck':4, 'Pedestrian': 1, 'Person_sitting': 1, 
    'Cyclist': 2, 'Tram': 0, 'Misc': 0, 'DontCare': 0}

NORM = 375.0 

df = pd.read_csv('annotations.csv')
new_df = df.loc[df['class'] != 'DontCare']
result_df = pd.DataFrame(columns=['filename','isperson','iscycle','iscar','istruck', \
                           'xmin', 'ymin', 'xmax', 'ymax', \
                           'angle', 'xloc', 'yloc', 'zloc'])

pbar = tqdm(total=new_df.shape[0], position=1)

for idx, row in new_df.iterrows():
    pbar.update(1)

    if os.path.exists(os.path.join("labels", row['filename']) and class_translation[row['class']] != 0):
        
        result_df.at[idx, 'filename'] = row['filename']

        ordclass = class_translation[row['class']]
        result_df.at[idx, 'isperson'] = 1.0 if ordclass == 1 else 0.0
        result_df.at[idx, 'iscycle'] = 1.0 if ordclass == 2 else 0.0
        result_df.at[idx, 'iscar'] = 1.0 if ordclass == 3 else 0.0
        result_df.at[idx, 'istruck'] = 1.0 if ordclass == 4 else 0.0

        xmin = float(row['xmin']) / NORM
        ymin = float(row['ymin']) / NORM
        xmax = float(row['xmax']) / NORM
        ymax = float(row['ymax']) / NORM
        dist = float(row['zloc']) / 100.0

        relw = abs(xmax - xmin)
        relh = abs(ymax - ymin)
        cx = (xmax+xmin) / 2.0
        cy = (ymax+ymin) / 2.0
        off = abs(0.5 - cx)


        result_df.at[idx, 'xmin'] = int(row['xmin'])
        result_df.at[idx, 'ymin'] = int(row['ymin'])
        result_df.at[idx, 'xmax'] = int(row['xmax'])
        result_df.at[idx, 'ymax'] = int(row['ymax'])

        result_df.at[idx, 'relw'] = relw
        result_df.at[idx, 'relh'] = relh
        result_df.at[idx, 'cx'] = cx
        result_df.at[idx, 'cy'] = cy
        result_df.at[idx, 'yoff'] = off
        result_df.at[idx, 'dist'] = dist

        result_df.at[idx, 'angle'] = row['observation angle']
        result_df.at[idx, 'xloc'] = int(row['xloc'])
        result_df.at[idx, 'yloc'] = int(row['yloc'])
        result_df.at[idx, 'zloc'] = int(row['zloc'])

mask = np.random.rand(len(result_df)) < 0.9
train = result_df[mask]
test = result_df[~mask]

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
