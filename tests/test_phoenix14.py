import unittest
from csi_sign_language.dataset.phoenix14 import VideoGenerator, Phoenix14Dataset
import glob
import numpy as np


class TestPhoenix14(unittest.TestCase):

    def test_VideoGenerator(self):
        video_dir = r'/home/jingyan/pycharm_remote/csi_sign_language_uni_laptop/dataset/phoenix2014-release/phoenix' \
                    r'-2014-multisigner/features/fullFrame-210x260px/dev/01April_2010_Thursday_heute_default-1/1/*.png'
        gen = VideoGenerator(glob.glob(video_dir))
        for item in gen:
            self.assertTupleEqual(item.shape, (260, 210, 3))

    def test_mediaPipe(self):
        video_dir = r'/home/jingyan/pycharm_remote/csi_sign_language_uni_laptop/dataset/phoenix2014-release/phoenix' \
                    r'-2014-multisigner/features/fullFrame-210x260px/dev/01April_2010_Thursday_heute_default-1/1/*.png'
        gen = VideoGenerator(glob.glob(video_dir))

    def test_phoenix14(self):
        data_root = r'/home/jingyan/pycharm_remote/csi_sign_language_home_win11/dataset/phoenix2014-release'
        dataset = Phoenix14Dataset(data_root)
        a = dataset[0]
        len(dataset)

if __name__ == '__main__':
    unittest.main()
