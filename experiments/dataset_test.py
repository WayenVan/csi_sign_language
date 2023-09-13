import sys
sys.path.append('src')
import os
from pathlib import Path
from csi_sign_language.data.dataset.phoenix14 import Phoenix14SegDataset, Phoenix14GraphSegDataset
import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from csi_sign_language.data.build import build_dataset
from csi_sign_language.utils.logger import build_logger


@hydra.main('../configs', 'defaultv2', version_base=None)
def main(cfg: DictConfig):
    logger = build_logger('name', 'experiments/log2.log')
    dataset = build_dataset(cfg)
    a = dataset['train_set'][0]['annotation']
    v = dataset['train_set'].dataset.reduced_vocab
    print(len(v))
    
    
main()