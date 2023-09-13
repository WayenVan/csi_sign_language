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

def stand(x, axis):
    mean = np.mean(x, axis, keepdims=True)
    std = np.std(x, axis, keepdims=True)
    return (x - mean)/ (std + 1e-7)

def norm(x, axis):
    mmax = np.max(x, axis, keepdims=True)
    mmin = np.min(x, axis, keepdims=True)
    return (x - mmin) / (mmax - mmin)

def interp(x: np.array, mask=None):
    """interp the signal of each channel in time
    :param x: [time, nodes, channel]
    :param mask: mask in time
    """
    num_nodes = x.shape[-2]
    channels = x.shape[-1]
    for node_idx in range(num_nodes):
        for channel_idx in range(channels):
            signal = x[:, node_idx, channel_idx] 
            no_zero_indices = np.where(signal != 0)[0]
            zero_indices = np.where(signal == 0)[0]
            if len(no_zero_indices) == 0:
                interp_values = signal[zero_indices]
            else:
                interp_values = np.interp(zero_indices, no_zero_indices, signal[no_zero_indices])
            x[zero_indices, node_idx, channel_idx] = interp_values
    
    if mask is not None:
        x[~mask, :, :] = 0
        
    return x

