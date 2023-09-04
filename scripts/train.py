from omegaconf import OmegaConf, DictConfig
import sys
sys.path.append('src')
from csi_sign_language.utils.logger import build_logger, strtime
from csi_sign_language.engines.build import build_trainner
from csi_sign_language.data.build import build_dataloader
from csi_sign_language.models.build_model import build_gnn_unet_model
import torch
import hydra
import os

@hydra.main(version_base=None, config_path='../configs', config_name='default.yaml')
def main(cfg: DictConfig):
    
    save_dir = os.path.join(cfg.save_dir, f'experiment-{strtime()}')
    os.makedirs(save_dir)
    
    data_loaders = build_dataloader(cfg)
    logger = build_logger('main', os.path.join(save_dir, 'train.log'))
    model = build_gnn_unet_model(cfg)
    
    opt = torch.optim.SGD(model.parameters(), lr=cfg.train.lr)
    trainer = build_trainner(cfg, model, opt, data_loaders['train_loader'], logger, save_dir)
    trainer.do_train()
if __name__ == '__main__':
    main()