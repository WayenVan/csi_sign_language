from .models import *
from omegaconf import OmegaConf, DictConfig

def build_gnn_unet_model(cfg: DictConfig):
    
    if cfg.model.type == 'v2':
        bert = tf.BertModel.from_pretrained('bert-base-uncased')
        return GNNUnetV2(
            cfg.model.d_model,
            cfg.model.in_channel,
            cfg.model.num_class,
            cfg.model.num_node_hand,
            cfg.model.num_node_pose,
            bert
        )
    