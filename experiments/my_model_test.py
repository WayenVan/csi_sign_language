import sys
sys.path.append('src')
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from csi_sign_language.models.models import GNNUnet
from csi_sign_language.data.build import build_dataloader
from omegaconf import OmegaConf
from tqdm import tqdm
cfg = OmegaConf.load('/home/jingyan/Documents/csi_sign_language/configs/config.yaml')
loader = build_dataloader(cfg)['train_loader']


model = GNNUnet(128, 2, 1, loader.dataset.dataset.NUM_CLASS, 21).to('cuda')
for data in tqdm(loader):
    lhand = data['lhand']
    lhand: torch.tensor = rearrange(lhand, 'b (tmp clip) n xy -> (b tmp) clip n xy', clip=32)
    lhand = lhand.type(torch.float32).to('cuda')
    edges = torch.tensor(loader.dataset.dataset.HAND_CONNECTION, dtype=torch.int64).to('cuda')
    out = model(lhand, edges)

    

    

    