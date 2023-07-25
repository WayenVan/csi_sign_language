import sys
sys.path.append('src')

from torch.utils.data import DataLoader
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from pathlib import Path
from csi_sign_language.dataset.phoenix14 import Phoenix14SegDatset, SegCollateGraph
import cv2
import networkx as nx

base_options = python.BaseOptions(model_asset_path='resources/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
    )
detector = vision.HandLandmarker.create_from_options(options)

phoenix_dir = os.path.join(Path(__file__).resolve().parent, '../dataset/phoenix2014-release')
# STEP 3: Load the input image.
dataset = Phoenix14SegDatset(phoenix_dir, length_time=320, length_glosses=40, padding_mode='back')
loader = DataLoader(dataset, collate_fn=SegCollateGraph(detector=detector, clip_size=32))

for item in loader:
    print(item[0].shape)
