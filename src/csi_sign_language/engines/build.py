from omegaconf import OmegaConf
from .trainers import Trainner
import numpy as np
import os

def build_trainner(
    cfg: OmegaConf,
    model,
    optimizer,
    loader,
    logger,
    save_dir
):
    hand_connection = np.load(
        os.path.join(cfg.data.graph_subset_root, 'hand_connection.npy')
    )
    pose_connection = np.load(
        os.path.join(cfg.data.graph_subset_root, 'pose_connection.npy')
    )
    return Trainner(
        model,
        optimizer,
        loader,
        logger,
        cfg.data.num_class,
        save_dir,
        cfg.device,
        hand_connection,
        pose_connection
    )