import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
from torchmetrics.classification import Accuracy
from tqdm import tqdm
from einops import rearrange
import logging


class Inferencer():
    
    def __init__(
            self,
            model: Module,
            loader: DataLoader,
            hand_connection,
            pose_connection,
            device,
            logger: logging.Logger,
            num_class,
            clip_size
        ) -> None:
        self.device=device
        self.model: Module = model
        self.loader = loader
        self.hand_connection = hand_connection
        self.pose_connection = pose_connection
        self.logger = logger.getChild(__class__.__name__)
        self.NUM_CLASS = num_class
        self.clip_size = clip_size
    
    def _rearrange_data(self, data):
        data: torch.tensor = rearrange(data, 'b (tmp clip) n xy -> (b tmp) clip n xy', clip=self.clip_size)
        data = data.type(torch.float32).to(self.device)
        return data 
        
    def do_inference(self):
        self.model.to(self.device)
        self.model.eval()
        self.logger.info('start inference')
        batch_accus = []
        accuracy = Accuracy(task='multiclass', num_classes=self.NUM_CLASS).to(self.device)
        batch_results = []
        for idx, data in enumerate(tqdm(self.loader)):
            annotation = data['annotation']
            lhand = data['lhand']
            rhand = data['rhand']
            pose = data['pose']
            b_size = lhand.size()[0]
            
            lhand, rhand, pose = tuple(map(self._rearrange_data, [lhand, rhand, pose]))
            hand_edges = torch.tensor(self.hand_connection, dtype=torch.int64).to(self.device)
            pose_edges = torch.tensor(self.pose_connection, dtype=torch.int64).to(self.device)

            with torch.no_grad():
                output = self.model(lhand, rhand, pose, hand_edges, pose_edges)
            output = rearrange(output, 'b s c -> (b s) c').to(self.device)
            annotation = rearrange(annotation, 'b s -> (b s)').to(self.device)

            accuracy.update(output, annotation)
            batch_accuracy = accuracy(output, annotation)
            batch_accus.append(batch_accuracy)
            self.logger.info(f'batch accuracy: {batch_accuracy}')

            predicted_label = torch.argmax(output, dim=-1)
            predicted_label = rearrange(predicted_label, '(b s) -> b s', b=b_size)
            batch_results.append(predicted_label)
        
        batch_results = torch.cat(batch_results, dim=0)
            
        return dict(
            accuracy=accuracy.compute(),
            predicted=batch_results.cpu().numpy(),
            batch_accuracy = batch_accuracy
        )
