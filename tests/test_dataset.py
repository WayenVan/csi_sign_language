import sys
sys.path.append('src')

from csi_sign_language.dataset.utils import holistic_recognition
from collections import OrderedDict
import pytest
from csi_sign_language.dataset.phoenix14 import *
import glob
import numpy as np
import mediapipe as mp
from pathlib import Path
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


# @pytest.fixture
def phoenix_dir():
    return os.path.join(Path(__file__).resolve().parent, '../dataset/phoenix2014-release')


def test_VideoGenerator(phoenix_dir):
    video_dir = r'phoenix-2014-multisigner/features/fullFrame-210x260px/dev/01April_2010_Thursday_heute_default-1/1/*.png'
    video_dir = os.path.join(phoenix_dir, video_dir)
    file_list = glob.glob(video_dir)
    file_list = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('-')[0][2:]))

    gen = VideoGenerator(file_list)
    for frame in gen:
        cv2.imshow('video', frame)
        cv2.waitKey(30)

def test_mediaPipe():
    mp_holistic = mp.solutions.holistic
    video_dir = r'/home/jingyan/pycharm_remote/csi_sign_language_uni_laptop/dataset/phoenix2014-release/phoenix' \
                r'-2014-multisigner/features/fullFrame-210x260px/dev/01April_2010_Thursday_heute_default-1/1/*.png'
    gen = VideoGenerator(glob.glob(video_dir))
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True) as holistic:

        e = enumerate(gen)
        _, frame = next(e)
        results = holistic_recognition(frame, holistic)
        print(results)

def test_phoenix14(phoenix_dir):
    data_root = phoenix_dir
    dataset = Phoenix14Dataset(data_root, length_time=500, length_glosses=40)
    
    _, (data, label) = next(enumerate(dataset))

    vocab = dataset.gloss_vocab.get_stoi()
    vocab = OrderedDict(sorted(vocab.items(), key=lambda x: x[1]))
    for i in range(len(vocab)):
        assert list(vocab.items())[i][1] == i

def test_phoenix14Seg(phoenix_dir):
    dataset = Phoenix14SegDatset(phoenix_dir, length_time=350, length_glosses=40, padding_mode='back')
    data, label, mask = dataset[200]

    vocab = dataset.frame_level_vocab.get_stoi()
    assert vocab['<PAD>'] == 0
    assert vocab['ABEND2'] == 12
    
    label_intext = dataset.frame_level_vocab.lookup_tokens(label)
    for idx, frame in enumerate(data):
        cv2.putText(frame, label_intext[idx], (0, 200), cv2.FONT_HERSHEY_DUPLEX, 1., (255,0,0), 1)
        cv2.imshow('video', frame)
        cv2.waitKey(30)
        
if __name__ == '__main__':
    test_phoenix14Seg(phoenix_dir())