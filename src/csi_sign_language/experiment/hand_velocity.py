from ..dataset.utils import VideoGenerator, holistic_recognition
import glob
import mediapipe as mp
import numpy as np
import collections


def main():
    data_dir = r'/home/jingyan/pycharm_remote/csi_sign_language_uni_laptop/dataset/phoenix2014-release/phoenix-2014' \
               r'-multisigner/features/fullFrame-210x260px/test/01April_2010_Thursday_heute_default-5/1/*.png'
    ring_buffer = [collections.deque(maxlen=10) for i in range(3)]
    shape = [(33, 3), (21, 2), (21, 2)]

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True) as holistic:

        video = VideoGenerator(glob.glob(data_dir))
        for frame in video:
            results = holistic_recognition(frame, holistic)




if __name__ == '__main__':
    main()
