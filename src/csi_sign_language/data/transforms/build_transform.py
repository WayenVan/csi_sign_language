from omegaconf import DictConfig, OmegaConf
from .transforms import *
import torchvision.transforms as T

def build_transform(cfg: DictConfig, type):
    
    if type=='img2keypoints':
        mediapipe_conf = cfg['mediapipe']
        return Image2Keypoints(mediapipe_conf['hand_asset'], mediapipe_conf['pose_asset'])
    
    
    