from omegaconf import OmegaConf, DictConfig
from .trainers import Trainner
from .inferencers import *
import numpy as np
import os

def _load_connections(cfg: DictConfig):
    
    hand_connection = np.load(
        os.path.join(cfg.data.graph_subset_root, 'hand_connection.npy')
    )
    pose_connection = np.load(
        os.path.join(cfg.data.graph_subset_root, 'pose_connection.npy')
    )

    return hand_connection, pose_connection
    
def build_trainner(
    cfg: DictConfig,
    model,
    optimizer,
    loader,
    logger,
    save_dir,
):
    hand_connection, pose_connection = _load_connections(cfg)
    return Trainner(
        model,
        optimizer,
        loader,
        logger,
        cfg.data.num_class,
        save_dir,
        cfg.device,
        hand_connection,
        pose_connection,
        cfg.data.clip_size
    )

def build_inferencer(
    cfg: DictConfig,
    model,
    loader,
    logger,
):
    hand_connection, pose_connection = _load_connections(cfg)
    inferencer = Inferencer(
        model,
        loader,
        hand_connection,
        pose_connection,
        cfg.device,
        logger,
        cfg.data.num_class,
        cfg.data.clip_size
    )
    return inferencer