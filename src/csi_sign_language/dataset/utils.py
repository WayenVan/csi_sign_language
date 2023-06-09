import numpy as np
import cv2
import typing
from ..csi_typing import PaddingMode

class VideoGenerator:

    def __init__(self, frame_list: typing.List[str]):
        self.__frame_list = frame_list

    def __iter__(self) -> np.ndarray:
        for file in self.__frame_list:
            yield cv2.imread(file)


def padding(data: np.ndarray, axis: int, length: int, padding_mode: PaddingMode):
    npad = [[0, 0] for i in data.shape]
    if padding_mode == 'front':
        npad[axis][0] = length - data.shape[axis]
    elif padding_mode == 'back':
        npad[axis][1] = length - data.shape[axis]
    else:
        raise Exception('padding_mode should be front or back')
    return np.pad(data, npad, mode='constant', constant_values=0)


