from .dataset.phoenix14 import Phoenix14GraphSegDataset
from torch.utils.data import Subset, DataLoader
from omegaconf import OmegaConf, DictConfig
from .transforms.build_transform import build_transform
import os

def build_dataset(cfg: DictConfig):
    trans = build_transform(cfg)
    dataset = Phoenix14GraphSegDataset(
        cfg.data.phoenix14_root, 
        cfg.data.graph_subset_root,
        cfg.data.time_length,
        transformation=trans
    )

    meta = OmegaConf.load(os.path.join(cfg.data.graph_subset_root, 'meta.yaml'))
    train_idx = meta.train_indexes
    val_idx = meta.val_indexes
    test_idx = meta.test_indexes
    
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    return dict(
        train_set = train_set,
        val_set = val_set,
        test_set = test_set
    )

def build_dataloader(cfg: DictConfig):
    dataset = build_dataset(cfg)
    b_train = cfg.data.train_loader.batch_size
    b_val = cfg.data.validate_loader.batch_size
    b_test = cfg.data.test_loader.batch_size
    return dict(
        train_loader = DataLoader(dataset['train_set'], batch_size=b_train, num_workers=cfg.data.num_workers),
        val_loader = DataLoader(dataset['val_set'], batch_size=b_val, num_workers=cfg.data.num_workers),
        test_loader = DataLoader(dataset['test_set'], batch_size=b_test, num_workers=cfg.data.num_workers)
    )

