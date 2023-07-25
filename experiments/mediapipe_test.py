import sys
sys.path.append('src')

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from pathlib import Path
from csi_sign_language.dataset.phoenix14 import Phoenix14SegDatset
from csi_sign_language.dataset.utils import hand_recognition
import cv2
import networkx as nx

phoenix_dir = os.path.join(Path(__file__).resolve().parent, '../dataset/phoenix2014-release')

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='resources/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
    )
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
dataset = Phoenix14SegDatset(phoenix_dir, length_time=350, length_glosses=40, padding_mode='back')
data, label, mask = dataset[0]

# STEP 4: Detect hand landmarks from the input image.
for image in data:
    ret = hand_recognition(image, detector)
    a = nx.get_node_attributes(ret['Left'], 'x')
    # image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_)
    # detection_result = detector.detect(mp_image)
    # ret = {}
    # assert len(detection_result.handedness) == len(detection_result.hand_landmarks)
    # for head, landmarks in list(zip(detection_result.handedness, detection_result.hand_landmarks)):
    #     G = nx.Graph()
    #     name = head[0].display_name
    #     for i, landmark in enumerate(landmarks):
    #         G.add_node(i, x=landmark.x, y=landmark.y)
    #     for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
    #         G.add_edge(c.start, c.end)

    #     ret[name] = G
    #     a = nx.get_node_attributes(G, 'x')

        
        
    
    