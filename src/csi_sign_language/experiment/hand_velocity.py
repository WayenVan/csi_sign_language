from ..dataset.utils import VideoGenerator
import glob
import numpy as np


def main():
    data_dir = r'/home/jingyan/pycharm_remote/csi_sign_language_uni_laptop/dataset/phoenix2014-release/phoenix-2014' \
               r'-multisigner/features/fullFrame-210x260px/test/01April_2010_Thursday_heute_default-5/1/*.png'

    video = VideoGenerator(glob.glob(data_dir))
    for frame in video:
        print(frame.shape)


if __name__ == '__main__':
    main()
