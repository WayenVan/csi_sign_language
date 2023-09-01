from typing import Any
from ...utils.mediapipe import MediapipeDetector
import numpy as np

class Image2Keypoints():
    
    def __init__(self, hand_asset, pose_asset) -> None:
        self.detector = MediapipeDetector(hand_asset, pose_asset)
        self.hand_shape = (21, 2)
        self.pose_shape = (33, 2)
    
    def __call__(self, data: dict) -> Any:
        assert 'img' in data, 'input data must contain img key'
        img: np.ndarray = data['img']
        hand_res = self.detector.hand_recognition(img)
        pose_res = self.detector.pose_recognition(img)
        
        lhand_f = False
        rhand_f = False
        pose_f = False
        
        if 'Right' in hand_res.keys():
            lhand = hand_res['Right']
            lhand_f = True
        else:
            lhand = np.zeros(self.hand_shape)
            
        if 'Left' in hand_res.keys():
            rhand = hand_res['Left']
            rhand_f = True
        else:
            rhand = np.zeros(self.hand_shape)
            
        if 'pose' in pose_res.keys():
            pose = pose_res['pose']
            pose_f = True
        else:
            pose = np.zeros(self.pose_shape)

        assert lhand.shape == rhand.shape == self.hand_shape
        assert pose.shape == self.pose_shape

        return dict(
            pose = pose,
            lhand = lhand,
            rhand = rhand,
            lhand_flag = lhand_f,
            rhand_flag = rhand_f,
            pose_flag = pose_f,
        ) 

