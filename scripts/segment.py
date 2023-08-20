import numpy as np
import os
import pandas as pd
import cv2
import sys
sys.path.append('src')
from csi_sign_language.dataset.utils import MediapipeDetector
import pathlib as pl


training_class_txt = '/home/jingyan/Documents/csi_sign_language/dataset/phoenix2014-release/phoenix-2014-multisigner/annotations/automatic/train.alignment'
path_to_features = '/home/jingyan/Documents/csi_sign_language/dataset/phoenix2014-release/phoenix-2014-multisigner'

hand_shape = (21, 2)
pose_shape = (33, 2)


save_dir = '/home/jingyan/Documents/csi_sign_language/dataset/graph_subset'
npy_dir = 'features'

alignment = pd.read_csv(training_class_txt, index_col=None, header=None, sep=" ")
frames = alignment.iloc[:, 0]
frame_info = pd.DataFrame(columns=['feature', 'lhand', 'rhand', 'pose'])
mp = MediapipeDetector()
for frame in frames:
    img = cv2.imread(os.path.join(path_to_features, frame))
    hand_res = mp.hand_recognition(img)
    pose_res = mp.pose_recognition(img)
    name = pl.PurePath(frame).parts[-1][:-4]
    
    if 'Right' in hand_res.keys():
        lhand = hand_res['Right']
    else:
        lhand = np.zeros(hand_shape)
        
    if 'Left' in hand_res.keys():
        rhand = hand_res['Left']
    else:
        rhand = np.zeros(hand_shape)
        
    if 'pose' in pose_res.keys():
        pose = pose_res['pose']
    else:
        pose = np.zeros(pose_shape)

    assert lhand.shape == rhand.shape == hand_shape
    assert pose.shape == pose_shape
            
    lhand_name = f'{name}_lhand.npy'
    rhand_name = f'{name}_rhand.npy'
    pose_name = f'{name}_pose.npy'

    np.save(os.path.join(save_dir, npy_dir, lhand_name), lhand)
    np.save(os.path.join(save_dir, npy_dir, rhand_name), rhand)
    np.save(os.path.join(save_dir, npy_dir, pose_name), pose)

    
    row = pd.Series([frame, lhand_name, rhand_name, pose_name], index=frame_info.columns)
    frame_info.loc[len(frame_info)] = row
    frame_info.to_csv(os.path.join(save_dir, 'annotations.csv'), index=False)
    continue