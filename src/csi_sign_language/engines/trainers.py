import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from logging import Logger
from tqdm import tqdm
from ..models.models import GNNUnet
from torchmetrics.classification import Accuracy
from einops import rearrange
import numpy as np

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
        pose_connection,
        clip_size
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
        self.clip_size = clip_size

    def _rearrange_data(self, data):
        data: torch.tensor = rearrange(data, 'b (tmp clip) n xy -> (b tmp) clip n xy', clip=self.clip_size)
        data = data.type(torch.float32).to(self.device)
        return data 

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
            rhand = data['rhand']
            pose = data['pose']
            
            lhand, rhand, pose = tuple(map(self._rearrange_data, [lhand, rhand, pose]))
            hand_edges = torch.tensor(self.hand_connection, dtype=torch.int64).to(self.device)
            pose_edges = torch.tensor(self.pose_connection, dtype=torch.int64).to(self.device)
            
            output = self.model(lhand, rhand, pose, hand_edges, pose_edges)
            output = rearrange(output, 'b s c -> (b s) c').to(self.device)
            annotation = rearrange(annotation, 'b s -> (b s)').to(self.device)
            loss = self.loss_fn(output, annotation)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            accuracy.update(output, annotation)
            batch_accuracy = accuracy(output, annotation)
            #calculate mask percentage
            mask = data['time_mask'].numpy()
            mask = rearrange(mask, 'b s -> (b s)')
            mask: np.ndarray = np.invert(mask)
            total_mask = np.sum(mask.astype(np.int8))
            percentage = total_mask/mask.shape[0]
            

            self.logger.info(f'iteration index: {idx}, batch loss: {loss.item()}, batch accuracy: {batch_accuracy}, padding percentage: {percentage}')
            batch_accus.append(batch_accuracy)
            batch_losses.append(loss.item())
        
        return dict(
            accuracy=accuracy.compute(),
            batch_losses=batch_losses,
            batch_accuracy = batch_accuracy
        )

            
        
        


            


