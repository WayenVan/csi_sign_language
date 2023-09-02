import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from logging import Logger
from tqdm import tqdm

class Trainner():
    
    def __init__(self) -> None:
        pass

    def do_train(self, model: Module, optimizer: Optimizer, train_loader: DataLoader, loss_fn, _logger: Logger ):
        model.train()
        logger = _logger.getChild(__class__.__name__)
        logger.info('start training')
        for data in tqdm(DataLoader):
            optimizer.zero_grad()
            x: torch.Tensor = torch.zeros(3,2)
            x.backward()
            optimizer.step()
            


