import sys
sys.path.append('src')
import os
from pathlib import Path
from csi_sign_language.data.dataset.phoenix14 import Phoenix14SegDataset, Phoenix14GraphSegDataset
import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from csi_sign_language.data.transforms.build_transform import build_transform

phoenix_dir = os.path.join(Path(__file__).resolve().parent, '../dataset/phoenix2014-release')
subset_dir = os.path.join(Path(__file__).resolve().parent, '../dataset/graph_subset')
dataset = Phoenix14GraphSegDataset(phoenix_dir, subset_dir, length_time=350, padding_mode='back')
# STEP 4: Detect hand landmarks from the input image.

@hydra.main('../configs', 'test', version_base=None)
def main(cfg: DictConfig):
    t = build_transform(cfg, type='img2keypoints')
    for i in tqdm.tqdm(range(len(dataset))):
        a = dataset[i]
main()