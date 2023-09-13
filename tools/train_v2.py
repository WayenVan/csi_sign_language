from omegaconf import OmegaConf, DictConfig
import sys
sys.path.append('src')
from csi_sign_language.utils.logger import build_logger, strtime
from csi_sign_language.engines.build import build_trainner, build_inferencer
from csi_sign_language.data.build import build_dataloader
from csi_sign_language.models.build_model import build_gnn_unet_model
import torch
import hydra
import os
import shutil

@hydra.main(version_base=None, config_path='../configs', config_name='defaultv2.yaml')
def main(cfg: DictConfig):
    script = os.path.abspath(__file__)
    save_dir = os.path.join(cfg.save_dir, f'experiment-{strtime()}')
    os.makedirs(save_dir)
    OmegaConf.save(cfg, os.path.join(save_dir, 'config.yaml'))
    logger = build_logger('main', os.path.join(save_dir, 'train.log'))
    shutil.copyfile(script, os.path.join(save_dir, 'script.py'))

    logger.info('building model and dataloaders')
    data_loaders = build_dataloader(cfg)
    model = build_gnn_unet_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    logger.info('building trainner and inferencer')
    trainer = build_trainner(cfg, model, opt, data_loaders['train_loader'], logger, save_dir)
    inferencer = build_inferencer(cfg, model, data_loaders['val_loader'], logger)

    logger.info('training loop start')
    best_accu = 0.
    for epoch in range(cfg.train.epoch):
        logger.info(f'epoch {epoch}')
        t_result = trainer.do_train()
        logger.info(f'finish one epoch')
        logger.info('start validation')
        result = inferencer.do_inference()
        accuracy = result['accuracy']
        logger.info(f'validate acuuracy {accuracy}')
        if accuracy >= best_accu:
            best_accu = accuracy
            logger.info('best model saving')
            torch.save(model, os.path.join(save_dir, 'model.pt'))
        
        
if __name__ == '__main__':
    main()