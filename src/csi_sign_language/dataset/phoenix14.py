import glob
from collections import Counter
import cv2 as cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import numpy as np

from csi_sign_language.csi_typing import PaddingMode
from .utils import VideoGenerator, padding
from .dictionary import Dictionary
from  typing import Tuple, List
from ..csi_typing import *
from torchtext.vocab import vocab, build_vocab_from_iterator, Vocab

from abc import ABC, abstractmethod

class BasePhoenix14Dataset(Dataset, ABC):
    
    def __init__(self, data_root, type='train', multisigner=True, length_time=None, length_glosses=None,
                padding_mode : PaddingMode ='front', gloss_dict: Vocab=None):
        if multisigner:
            annotation_dir = os.path.join(data_root, 'phoenix-2014-multisigner/annotations/manual')
            annotation_file = type + '.corpus.csv'
            feature_dir = os.path.join('phoenix-2014-multisigner/features/fullFrame-210x260px', type)
        else:
            annotation_dir = os.path.join(data_root, 'phoenix-2014-signerindependent-SI5/annotations/manual')
            annotation_file = type + '.SI5.corpus.csv'
            feature_dir = os.path.join('phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px', type)

        self._annotations = pd.read_csv(os.path.join(annotation_dir, annotation_file), delimiter='|')
        self._feature_dir = feature_dir
        self._data_root = data_root

        self._length_time = self.max_length_time if length_time == None else length_time
        self._length_gloss = self.max_length_gloss if length_glosses == None else length_glosses
        self._padding_mode = padding_mode

        self.gloss_vocab = self._create_glossdictionary() if gloss_dict == None else gloss_dict

    def __len__(self):
        return len(self._annotations)

    @abstractmethod
    def __getitem__(self, idx):
        return

    def _create_glossdictionary(self):
        def tokens():
            for annotation in self._annotations['annotation']:
                yield annotation.split()
        vocab = build_vocab_from_iterator(tokens(), special_first=True, specials=['<PAD>'])
        return vocab

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
    

class Phoenix14Dataset(BasePhoenix14Dataset):
    """
    Dataset for general RGB image with gloss label, the output is (frames, gloss_labels)
    """
    def __init__(self, data_root, type='train', multisigner=True, length_time=None, length_glosses=None, padding_mode: PaddingMode = 'front', gloss_dict=None, img_transform=None):
        super().__init__(data_root, type, multisigner, length_time, length_glosses, padding_mode, gloss_dict)
        self._img_transform = img_transform

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        anno = self._annotations['annotation'].iloc[idx]
        anno: List[str] = anno.split()
        anno: List[int] = self.gloss_vocab(anno)
        anno: np.ndarray = np.asarray(anno)

        folder: str = self._annotations['folder'].iloc[idx]
        frame_files: List[str] = self._get_frame_file_list_from_annotation(folder)

        video_gen: VideoGenerator = VideoGenerator(frame_files)
        frames: List[np.ndarray] = [frame if self._img_transform == None else self._transform(frame)  for frame in video_gen]
        # [t, h, w, c]
        frames: np.ndarray = np.stack(frames)

        # padding
        frames = padding(frames, 0, self._length_time, self._padding_mode)
        anno = padding(anno, 0, self._length_gloss, self._padding_mode)
        return frames, anno
    
    def _get_frame_file_list_from_annotation(self, folder: str) -> List[str]:
        """return frame file list with the frame order"""
        file_list: List[str] = glob.glob(os.path.join(self._data_root, self._feature_dir, folder))
        file_list = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('-')[0][2:]))
        return file_list
        


class Phoenix14SegDatset(Phoenix14Dataset):
    """
    Dataset for frame level segmentation, the output is (frames, frames_level_annotation) 

    """
    def __init__(self, data_root, type='train', length_time=None, length_glosses=None, padding_mode: PaddingMode = 'front', gloss_dict=None, img_transform=None):
        super().__init__(data_root, type, True, length_time, length_glosses, padding_mode, gloss_dict, img_transform)
        self._frame_level_annotations_relative_path: str = 'phoenix-2014-multisigner/annotations/automatic'

        self._path_to_training_classes_txt: str= os.path.join(data_root, self._frame_level_annotations_relative_path, 'trainingClasses.txt')
        self._traning_classes: pd.DataFrame = pd.read_csv(self._path_to_training_classes_txt)
        self._frame_level_vocab: Vocab = build_vocab_from_iterator(iterator=self._traning_classes['signstate classlabel'])
    
    def __getitem__(self, idx):
        
        return super().__getitem__(idx)