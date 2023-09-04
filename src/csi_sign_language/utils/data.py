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
            
def assert_lists_same_length(lists):
    length = len(lists[0])
    assert all(len(lst) == length for lst in lists[1:]), "Lists have different lengths"

def padding(data: np.ndarray, axis: int, length: int, padding_mode: PaddingMode):

    delta_legnth = length - data.shape[axis]
    if delta_legnth <=0:
        return np.take(data, range(length), axis=axis), np.ma.make_mask(np.ones(length))
        

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

