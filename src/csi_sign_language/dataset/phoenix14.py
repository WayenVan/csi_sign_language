import cv2

class VideoGenerator:

    def __init__(self, frame_list):
        self.__frame_list = frame_list

    def __iter__(self):
        for file in self.__frame_list:
            yield cv2.imread(file)

