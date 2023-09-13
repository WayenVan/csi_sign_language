from omegaconf import DictConfig, OmegaConf
from .transforms import *
import torchvision.transforms as T 
import torch
from torchvision.transforms import Compose

def build_transform(cfg: DictConfig):
    return Compose([
        TimporalInterp(),
        TimporalNorm(),
    ])
    