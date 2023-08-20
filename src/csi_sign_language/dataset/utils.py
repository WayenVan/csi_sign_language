import numpy as np
import cv2
import typing
from ..csi_typing import PaddingMode
import mediapipe as mp
import networkx as nx
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

class VideoGenerator:

    def __init__(self, frame_list: typing.List[str]):
        self.__frame_list = frame_list

    def __iter__(self) -> np.ndarray:
        for file in self.__frame_list:
            yield cv2.imread(file)

def padding(data: np.ndarray, axis: int, length: int, padding_mode: PaddingMode):

    delta_legnth = length - data.shape[axis]

    npad = [[0, 0] for i in data.shape]
    if padding_mode == 'front':
        npad[axis][0] = delta_legnth
        mask = np.ones(length)
        mask[:delta_legnth] = 0
    elif padding_mode == 'back':
        npad[axis][1] = delta_legnth
        mask = np.ones(length)
        mask[-delta_legnth:] = 0
    else:
        raise Exception('padding_mode should be front or back')
    return np.pad(data, npad, mode='constant', constant_values=0), np.ma.make_mask(mask)


def holistic_recognition(image: np.ndarray, mp_solution: mp.solutions.holistic.Holistic):
    """
    :return: pose_landmarks[points, position(x, y visibility)],
    left_hand_landmarks[points, position(x, y)],
    right_hand_landmarks[points, position(x, y]
    """
    results = mp_solution.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks is not None:
        pose_landmarks = np.asarray([[point.x, point.y, point.visibility] for point in results.pose_landmarks.landmark])
    else:
        pose_landmarks = None

    if results.left_hand_landmarks is not None:
        left_hand_landmarks = np.asarray([[point.x, point.y] for point in results.left_hand_landmarks.landmark])
    else:
        left_hand_landmarks = None

    if results.right_hand_landmarks is not None:
        right_hand_landmarks = np.asarray([[point.x, point.y] for point in results.right_hand_landmarks.landmark])
    else:
        right_hand_landmarks = None

    return pose_landmarks, left_hand_landmarks, right_hand_landmarks


def hand_recognition(image: np.ndarray, detector) -> nx.Graph:
    image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_)
    detection_result = detector.detect(mp_image)
    ret = {}
    assert len(detection_result.handedness) == len(detection_result.hand_landmarks)
    for head, landmarks in list(zip(detection_result.handedness, detection_result.hand_landmarks)):
        G = nx.Graph()
        name = head[0].display_name
        for i, landmark in enumerate(landmarks):
            G.add_node(i, x=landmark.x, y=landmark.y)
        for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            G.add_edge(c.start, c.end)

        ret[name] = G
    return ret

class MediapipeDetector():
    
    def __init__(self) -> None:
        hand_opt = python.BaseOptions(model_asset_path='resources/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=hand_opt,
            num_hands=2
            )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)

        pose_opt = python.BaseOptions(model_asset_path='resources/pose_landmarker_lite.task')
        options = vision.PoseLandmarkerOptions(
            base_options=pose_opt,
            output_segmentation_masks=True)
        self.pose_detector = vision.PoseLandmarker.create_from_options(options)

        self.HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
        self.POSE_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS
    
    def hand_recognition(self, image: np.ndarray):
        """
        see https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#get_started
        """
        image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_)
        detection_result = self.hand_detector.detect(mp_image)
        assert len(detection_result.handedness) == len(detection_result.hand_landmarks)
        ret = {}
        for head, landmarks in list(zip(detection_result.handedness, detection_result.hand_landmarks)):
            name = head[0].display_name
            landmarks_np = []
            for landmark in landmarks:
                landmarks_np.append(np.array([landmark.x, landmark.y]))
            ret[name] = np.stack(landmarks_np)
        
        return ret

    def pose_recognition(self, image: np.ndarray):
        image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_)
        detection_result = self.pose_detector.detect(mp_image)
        ret = {}
        landmarks_np = []
        for landmark in detection_result.pose_landmarks[0]:
            landmarks_np.append(np.array([landmark.x, landmark.y]))
        ret['pose'] = np.stack(landmarks_np)
        return ret
