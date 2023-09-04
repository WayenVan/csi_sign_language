import sys
sys.path.append('src')
from omegaconf import OmegaConf
from csi_sign_language.data.build import build_dataloader
from tqdm import tqdm

def test_graphsetgment_dataset():
    """testing if all the dataloader is correct given a default config file
    """
    cfg = OmegaConf.load('configs/config.yaml') 
    loader = build_dataloader(cfg)
    for data in tqdm(loader['train_loader']):
        pass
    
    for data in tqdm(loader['test_loader']):
        pass

    for data in tqdm(loader['val_loader']):
        pass

    