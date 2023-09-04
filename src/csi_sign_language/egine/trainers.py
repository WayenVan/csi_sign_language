from omegaconf import DictConfig, OmegaConf
import tqdm
from torch.optim import Optimizer


class Trainer():
    
    def __init__(self, dataloader, model, loss_fn, optimizer: Optimizer) -> None:
        self.dataloader = dataloader
        self.opt = optimizer
        self.model = model
        self.loss_fn = loss_fn
    
    def do_train(self):
        for batched_data in tqdm.tqdm(self.dataloader):
            annotation = batched_data['annotation']
            lhand = batched_data['lhand']
            rhand = batched_data['rhand']
            time_mask = batched_data['time_mask']
            model()

            self.opt.zero_grad()
            loss = self.loss_fn()


            
            
            
        