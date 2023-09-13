import sys
sys.path.append('src')
import os
from pathlib import Path
from csi_sign_language.data.dataset.phoenix14 import Phoenix14SegDataset, Phoenix14GraphSegDataset
import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from csi_sign_language.data.build import build_dataset


@hydra.main('../configs', 'defaultv2', version_base=None)
def main(cfg: DictConfig):
    dataset = build_dataset(cfg)
    