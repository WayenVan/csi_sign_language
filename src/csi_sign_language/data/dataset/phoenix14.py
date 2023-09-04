import os
import numpy as np
import glob
import cv2 as cv2
import pandas as pd
from typing import Any, Tuple, List
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchtext.vocab import vocab, build_vocab_from_iterator, Vocab

from einops import rearrange

from csi_sign_language.csi_typing import PaddingMode
from ...csi_typing import *
from ...utils.data import VideoGenerator, padding, assert_lists_same_length

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
    Dataset for general RGB image with gloss label, the output is (frames, gloss_labels, frames_padding_mask, gloss_padding_mask)
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
        frames, frames_mask = padding(frames, 0, self._length_time, self._padding_mode)
        anno, anno_mask = padding(anno, 0, self._length_gloss, self._padding_mode)
        
        return frames, anno, frames_mask, anno_mask
    
    def _get_frame_file_list_from_annotation(self, folder: str) -> List[str]:
        """return frame file list with the frame order"""
        file_list: List[str] = glob.glob(os.path.join(self._data_root, self._feature_dir, folder))
        file_list = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('-')[0][2:]))
        return file_list
        

class Phoenix14SegDataset(Phoenix14Dataset):
    """
    Dataset for frame level segmentation, only multi train and multisigner is avialiable
    the output is (frames, frames_level_annotation) 
    special token added: <PAD>, so all classes index will +1 !!!!!

    """
    def __init__(self, data_root, length_time=None, padding_mode: PaddingMode = 'front', img_transform=None):
        super().__init__(data_root, 'train', True, length_time, None, padding_mode, None, img_transform)
        self._frame_level_annotations_relative_path: str = 'phoenix-2014-multisigner/annotations/automatic'
        self._path_to_training_classes_txt: str= os.path.join(data_root, self._frame_level_annotations_relative_path, 'trainingClasses.txt')
        self._path_to_alignment: str = os.path.join(data_root, self._frame_level_annotations_relative_path, 'train.alignment')
        
        self.frame_level_vocab = self._create_vocab()
        self.alignment = pd.read_csv(self._path_to_alignment, index_col=0, header=None, sep=" ")
        self.no_sign_token = 'si'

    
    def __getitem__(self, idx):
        
        folder: str = self._annotations['folder'].iloc[idx]
        frame_files: List[str] = self._get_frame_file_list_from_annotation(folder)
        frame_files_relative: List[str] = [self._remove_root_dir_from_directory(dir) for dir in frame_files]

        # read label into numpy
        anno_frame_levels = [self.alignment.loc[frame].item() + 1 for frame in frame_files_relative]
        
        # read video frames from super
        frames, _, frames_mask, _ = super().__getitem__(idx)
        
        # padding frame level annotations
        anno_frame_levels, anno_mask = padding(np.array(anno_frame_levels), 0, self._length_time, self._padding_mode)
        assert np.array_equal(frames_mask, anno_mask)

        mask = frames_mask
        return frames, anno_frame_levels, mask
        

    def _create_vocab(self):
        _training_classes: pd.DataFrame = pd.read_csv(self._path_to_training_classes_txt, delimiter=' ')
        class_dict: List[Tuple[str, int]] = [(row['signstate'], 1) \
                                            for index, row in _training_classes.iterrows()]
        class_dict: OrderedDict = OrderedDict(class_dict)
        return vocab(class_dict, specials=['<PAD>'])
    
    def _remove_root_dir_from_directory(self, dir: str):
        root_dirs = Path(self._data_root)
        target_dirs = Path(dir)
        ret = target_dirs.relative_to(root_dirs)
        ret = Path(*ret.parts[1:])
        return str(ret)

class Phoenix14GraphSegDataset(Phoenix14SegDataset):
    """
    Dataset to load the extracted keypoints subset from frame-level annotation.
    the output is (frames, frames_level_annotation) 
    special token added: <PAD>, so all classes index will +1 !!!!!
    """
    def __init__(self, phoenix14_data_root, subset_data_root, length_time=None, padding_mode: PaddingMode = 'front', img_transform=None):
        super().__init__(phoenix14_data_root, length_time, padding_mode, img_transform)
        self.subset_data_root = subset_data_root
        
        annotation_path = os.path.join(subset_data_root, 'annotations.csv')
        self.keypoints_annotation = pd.read_csv(annotation_path)
        self.keypoints_annotation = self.keypoints_annotation.set_index(self.keypoints_annotation['feature'])
        self.feature_root = os.path.join(subset_data_root, 'features')

        self.HAND_CONNECTION: np.ndarray = np.load(os.path.join(self.subset_data_root, 'hand_connection.npy'))
        self.POSE_CONNECTION: np.ndarray = np.load(os.path.join(self.subset_data_root, 'pose_connection.npy'))


    
    def __getitem__(self, idx):
        folder: str = self._annotations['folder'].iloc[idx]
        frame_files: List[str] = self._get_frame_file_list_from_annotation(folder)
        frame_files_relative: List[str] = [self._remove_root_dir_from_directory(dir) for dir in frame_files]

        anno_frame_levels = []
        lhand_array = []
        rhand_array = []
        pose_array = []
        for frame in frame_files_relative:
            # read label into numpy
            anno_frame_levels.append(self.alignment.loc[frame].item() + 1)

            # read label into
            feature_row = self.keypoints_annotation.loc[frame]

            lhand_file = os.path.join(self.feature_root, feature_row['lhand'])
            rhand_file = os.path.join(self.feature_root, feature_row['rhand'])
            pose_file = os.path.join(self.feature_root, feature_row['pose'])
        
            lhand_array.append(np.load(lhand_file))
            rhand_array.append(np.load(rhand_file))
            pose_array.append(np.load(pose_file))
        
        assert_lists_same_length([anno_frame_levels, lhand_array, rhand_array, pose_array])
        anno_frame_levels, lhand_array, rhand_array, pose_array = (np.stack(item) for item in (anno_frame_levels, lhand_array, rhand_array, pose_array))

        anno_frame_levels, mask = padding(anno_frame_levels, 0, self._length_time, self._padding_mode)
        lhand_array, _ = padding(lhand_array, 0, self._length_time, self._padding_mode)
        rhand_array , _ = padding(rhand_array, 0, self._length_time, self._padding_mode)
        pose_array, _ = padding(pose_array, 0, self._length_time, self._padding_mode)
        
        return dict(annotation=anno_frame_levels, lhand=lhand_array, rhand=rhand_array, pose=pose_array, time_mask=mask)