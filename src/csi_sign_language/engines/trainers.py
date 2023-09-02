import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from logging import Logger
from tqdm import tqdm
from ..models.models import GNNUnet
from torchmetrics.classification import Accuracy

class Trainner():
    
    def __init__(
        self, 
        model: GNNUnet, 
        optimizer: Optimizer, 
        train_loader: DataLoader, 
        loss_fn, 
        _logger: Logger
        ) -> None:

        self.model = model
        self.opt = optimizer
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.logger = _logger.getChild(__class__.__name__)

    def do_train(self):
        self.model.train()
        self.logger.info('start training')
        accuracy = Accuracy(task='multiclass', num_class=)
        for data in tqdm(DataLoader):
            self.opt.zero_grad()
            x: torch.Tensor = torch.zeros(3,2)
            x.backward()
            self.opt.step()
            


