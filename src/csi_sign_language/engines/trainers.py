import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from logging import Logger
from tqdm import tqdm
from ..models.models import GNNUnet
from torchmetrics.classification import Accuracy
from einops import rearrange

class Trainner():
    
    def __init__(
        self, 
        model: GNNUnet, 
        optimizer: Optimizer, 
        train_loader: DataLoader, 
        _logger: Logger,
        num_class: int,
        save_directory: str,
        device,
        hand_connection,
        pose_connection
        ) -> None:

        self.model = model
        self.opt = optimizer
        self.train_loader = train_loader
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.logger: Logger = _logger.getChild(__class__.__name__)
        self.NUM_CLASS = int(num_class)
        self.device = device
        self.hand_connection = hand_connection
        self.pose_connection = pose_connection
        self.save_directory = save_directory

    def do_train(self):
        self.model.to(self.device)
        self.model.train()
        self.logger.info('start training')
        
        batch_accus = []
        batch_losses = []
        
        accuracy = Accuracy(task='multiclass', num_classes=self.NUM_CLASS).to(self.device)
        for idx, data in enumerate(tqdm(self.train_loader)):
            annotation = data['annotation']
            lhand = data['lhand']
            lhand: torch.tensor = rearrange(lhand, 'b (tmp clip) n xy -> (b tmp) clip n xy', clip=32)
            lhand = lhand.type(torch.float32).to(self.device)
            edges = torch.tensor(self.hand_connection, dtype=torch.int64).to(self.device)
            
            output = self.model(lhand, edges)
            output = rearrange(output, 'b s c -> (b s) c').to(self.device)
            annotation = rearrange(annotation, 'b s -> (b s)').to(self.device)
            loss = self.loss_fn(output, annotation)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            accuracy.update(output, annotation)
            batch_accuracy = accuracy(output, annotation)
            self.logger.info(f'iteration index:{idx}, batch loss: {loss.item()}, batch accuracy: {batch_accuracy}')
            batch_accus.append(batch_accuracy)
            batch_losses.append(batch_losses)
        
        return dict(
            accuracy=accuracy.compute(),
            batch_losses=batch_losses,
            batch_accuracy = batch_accuracy
        )

            
        
        


            


