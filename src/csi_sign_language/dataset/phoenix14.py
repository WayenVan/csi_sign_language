import glob

import cv2 as cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import numpy as np

class VideoGenerator:

    def __init__(self, frame_list):
        self.__frame_list = frame_list

    def __iter__(self):
        for file in self.__frame_list:
            yield cv2.imread(file)


class Phoenix14Dataset(Dataset):

    def __init__(self, data_root, type='train', multisigner=True, length_time=None, length_glosses=None, padding_mode='front'):
        if multisigner:
            annotation_dir = os.path.join(data_root, 'phoenix-2014-multisigner/annotations/manual')
            annotation_file = type+'.corpus.csv'
            feature_dir = os.path.join('phoenix-2014-multisigner/features/fullFrame-210x260px', type)
        else:
            annotation_dir = os.path.join(data_root, 'phoenix-2014-signerindependent-SI5/annotations/manual')
            annotation_file = type+'.SI5.corpus.csv'
            feature_dir = os.path.join('phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px', type)

        self._annotations = pd.read_csv(os.path.join(annotation_dir, annotation_file), delimiter='|')
        self._feature_dir = feature_dir
        self._data_root = data_root
        self._length_time = length_time
        self._length_gloss = length_glosses
        self._padding_mode = padding_mode

    def __len__(self):
        return len(self._annotations)

    def __getitem__(self, idx):
        anno = self._annotations['annotation'].iloc[idx]
        folder = self._annotations['folder'].iloc[idx]
        file_list = glob.glob(os.path.join(self._data_root, self._feature_dir, folder))
        file_list = sorted(file_list, key=lambda x : int(x.split('_')[-1].split('-')[0][2:]))
        video_gen = VideoGenerator(file_list)
        frames = [frame for frame in video_gen]
        # [t, h, w, c]
        frames = np.stack(frames)

        #padding

    def _padding(self):
        pass

    @property
    def max_length_time(self):
        max = 0
        for folder in self._annotations['folder']:
            file_list = glob.glob(os.path.join(self._data_root, self._feature_dir, folder))
            if len(file_list) >= max:
                max = len(file_list)
        return max

    @property
    def max_length_gloss(self):
        max = 0
        for glosses in self._annotations['annotation']:
            l = len(glosses.split())
            if l > max:
                max = l
        return max