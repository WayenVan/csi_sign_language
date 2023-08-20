import sys
sys.path.append('src')
import os
from pathlib import Path
from csi_sign_language.dataset.phoenix14 import Phoenix14SegDatset
from csi_sign_language.dataset.utils import hand_recognition, MediapipeDetector
import cv2
import networkx as nx

mp_detector = MediapipeDetector()


phoenix_dir = os.path.join(Path(__file__).resolve().parent, '../dataset/phoenix2014-release')

dataset = Phoenix14SegDatset(phoenix_dir, length_time=350, length_glosses=40, padding_mode='back')
data, label, mask = dataset[0]

# STEP 4: Detect hand landmarks from the input image.
for image in data:
    mp_detector.hand_recognition(image)
    mp_detector.pose_recognition(image)
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

        
        
    
    