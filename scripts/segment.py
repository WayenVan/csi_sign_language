import numpy as np
import os
import pandas as pd
import cv2
import sys
sys.path.append('src')
from csi_sign_language.utils.mediapipe_tools import MediapipeDetector
import pathlib as pl
import tqdm

if_load = True
training_class_txt = '/home/jingyan/Documents/csi_sign_language/dataset/phoenix2014-release/phoenix-2014-multisigner/annotations/automatic/train.alignment'
path_to_features = '/home/jingyan/Documents/csi_sign_language/dataset/phoenix2014-release/phoenix-2014-multisigner'

hand_shape = (21, 2)
pose_shape = (33, 2)
hand_asset = '/home/wayenvan/Desktop/csi_sign_language/resources/hand_landmarker.task'
pose_asset = '/home/wayenvan/Desktop/csi_sign_language/resources/pose_landmarker_lite.task'


save_dir = '/home/wayenvan/Desktop/csi_sign_language/dataset/graph_subset'
npy_dir = 'features'

def segment():
    alignment = pd.read_csv(training_class_txt, index_col=None, header=None, sep=" ")
    frames = alignment.iloc[:, 0]

    if if_load:
        frame_info = pd.read_csv(os.path.join(save_dir, 'annotations.csv'), index_col=False)
        annotation_io = open(os.path.join(save_dir, 'annotations.csv'), 'a')
    else:
        frame_info = pd.DataFrame(columns=['feature', 'lhand', 'rhand', 'pose'])
        frame_info.to_csv(os.path.join(save_dir, 'annotations.csv'), index=False)
        annotation_io = open(os.path.join(save_dir, 'annotations.csv'), 'a')


    previous_frame_info = frame_info['feature']
    mp = MediapipeDetector(hand_asset, pose_asset)
    for frame in tqdm.tqdm(frames):
        if if_load and previous_frame_info.isin([frame]).any():
            print('exist, jumping')
            continue
        
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

        if if_load:
            row = f'{frame},{lhand_name},{rhand_name},{pose_name}\n'
            try:
                annotation_io.write(row)
            except KeyError as e:
                annotation_io.close()
                exit()
        

    annotation_io.close()
    
def generate_config():
    mp = MediapipeDetector(hand_asset=hand_asset, pose_asset=pose_asset)
    hand_connection = np.array([(item.start, item.end) for item in mp.HAND_CONNECTIONS])
    pose_connection = np.array([(item.start, item.end) for item in mp.POSE_CONNECTIONS])

    np.save(os.path.join(save_dir, 'hand_connection.npy'), hand_connection.transpose())
    np.save(os.path.join(save_dir, 'pose_connection.npy'), pose_connection.transpose())
    
    
    
    
if __name__ == '__main__':
    generate_config()