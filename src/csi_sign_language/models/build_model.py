from .models import *
from omegaconf import OmegaConf, DictConfig

def build_gnn_unet_model(cfg: DictConfig):
    return GNNUnet(
        cfg.model.d_model,
        cfg.model.in_channel,
        cfg.model.num_encoder,
        cfg.data.num_class,
        cfg.model.num_node
    )
    