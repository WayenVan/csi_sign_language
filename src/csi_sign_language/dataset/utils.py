import numpy as np
import cv2
import typing
from ..csi_typing import PaddingMode
import mediapipe as mp

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

