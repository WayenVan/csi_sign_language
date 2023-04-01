#! /usr/bin/env python3

from csi_sign_language.dataset.utils import VideoGenerator, holistic_recognition
import glob
import mediapipe as mp
import numpy as np
import collections
from typing import Tuple, List
import matplotlib.pyplot as plt
import scipy.signal as s

class VelocityCalculator:
    """calculate the velocity of keypoints recognized by mediapipe holistic
    """
    def __init__(self, buffer_size=10, time_gap=1.):
        self._ring_buffer: List[collections.deque] = [collections.deque(maxlen=10) for i in range(3)]
        self._shape = ((33, 2), (21, 2), (21, 2))
        self._time_gap = time_gap

    def calculate_velocity(self, results: Tuple[np.ndarray]) -> Tuple[np.ndarray, ...]:
        """

        :param results: results: (pose keypoints, left hand keypoint, right hand keypoints), they are all np.ndarray
            where: pose keypoints (33, 2),
                   lef hand keypoints (21, 2),
                   right hand keypoints (21, 2)
        :return (pose velocity, left hand velocity, right hand velocity)
        """
        ret = []
        for idx, result in enumerate(results):

            # if not recognized
            if result is None:
                ret.append(np.zeros(self._shape[idx]))
                continue

            if len(self._ring_buffer[idx]) == 0:
                self._ring_buffer[idx].append(result)
                ret.append(np.zeros(self._shape[idx]))
                continue

            prev = self._ring_buffer[idx][-1]
            v = result - prev / self._time_gap

            if idx == 0:
                v = v[:, :2]
            ret.append(v)

        return tuple(ret)
            

def main():
    data_dir =r'dataset/phoenix2014-release/phoenix-2014-multisigner/features/' \
        r'fullFrame-210x260px/test/01April_2010_Thursday_heute_default-5/1/*.png'
    calculator = VelocityCalculator()
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True) as holistic:
        video = VideoGenerator(glob.glob(data_dir))
        velocity = []
        for frame in video:
            results = holistic_recognition(frame, holistic)
            v = calculator.calculate_velocity(results)
            velocity.append(v)

        velocity = list(zip(*velocity))
        v_p = np.stack(velocity[0])
        v_l = np.stack(velocity[1])
        v_r = np.stack(velocity[2])

        plt.figure(figsize=(10, 6))
        # for landmark in (20, 22, 18, 16, 14 ,12, 24, 23 ,11, 13, 21, 15, 19, 17):
        for landmark in (21, 22, 17, 19):
            x = v_p[:, landmark, 0]
            y = v_p[:, landmark, 1]
            vvv = np.sqrt(x**2, y**2)
            vvv = s.medfilt(vvv, 9)
            plt.plot(vvv, linestyle='-', label=str(landmark))

        plt.legend()
        plt.title('medfiltered velocity changing')
        plt.xlabel('frame')
        plt.ylabel('normalized velocity')
        plt.ylim(-0.02, .2)
        # plt.savefig('results/experiment/velocity_filtered.pdf')
        plt.savefig('results/experiment/velocity_22-19.pdf')

if __name__ == '__main__':
    main()
