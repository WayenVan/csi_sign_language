import unittest
from csi_sign_language.dataset.phoenix14 import VideoGenerator, Phoenix14Dataset
from csi_sign_language.dataset.utils import holistic_recognition
import glob
import numpy as np
import mediapipe as mp

from torch.utils.data import DataLoader


class TestPhoenix14(unittest.TestCase):

    def test_VideoGenerator(self):
        video_dir = r'/home/jingyan/pycharm_remote/csi_sign_language_uni_laptop/dataset/phoenix2014-release/phoenix' \
                    r'-2014-multisigner/features/fullFrame-210x260px/dev/01April_2010_Thursday_heute_default-1/1/*.png'
        gen = VideoGenerator(glob.glob(video_dir))
        for item in gen:
            self.assertTupleEqual(item.shape, (260, 210, 3))

    def test_mediaPipe(self):
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

    def test_phoenix14(self):
        data_root = r'/home/jingyan/pycharm_remote/csi_sign_language_home_win11/dataset/phoenix2014-release'
        dataset = Phoenix14Dataset(data_root, length_time=500, length_glosses=40)
        
        _, data = next(enumerate(dataset))
        self.assertEqual(data[0].shape[0], 500)
        self.assertEqual(data[1].shape[0], 40)
        
        loader = DataLoader(dataset, batch_size=32)
        _, data = next(enumerate(dataset))
        print(data)

if __name__ == '__main__':
    unittest.main()
