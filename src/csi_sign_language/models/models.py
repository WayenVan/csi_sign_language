import torch
import torch.nn as nn
from einops import rearrange
import transformers as tf

from ..modules.unet import *
from ..modules.graph import *
    
class GNNUnetV2(nn.Module):

    def __init__(self, d_model, in_channel, n_classes, num_node_hand, num_node_pose, bert: tf.BertModel) -> None:
        super().__init__() 
        
        self.d_model = d_model
        
        bert_layers = bert.encoder.layer
        self.encoder_lhand = self.create_encoder(d_model, in_channel, num_node_hand, bert_layers)
        self.encoder_pose = self.create_encoder(d_model, in_channel, num_node_pose, bert_layers)
        self.encoder_rhand = self.create_encoder(d_model, in_channel, num_node_hand, bert_layers)
        
        self.pooling = nn.AdaptiveAvgPool2d((1, d_model))
        self.decoder = Unet1d(d_model*3, n_classes)
    
    @staticmethod
    def create_encoder(d_model, in_channel, num_node, bert_layers: nn.ModuleList):
        module_list = []
        for idx, bert_layer in enumerate(bert_layers):
            if idx == 0:
                module_list.append(GCNBertLayer(d_model, in_channel, d_model, num_node, bert_layer))
            else:
                module_list.append(GCNBertLayer(d_model, d_model, d_model, num_node, bert_layer))
        
        return nn.ModuleList(module_list)
                
    @staticmethod
    def encoder_forward(x, edges, encoder: nn.ModuleList):
        for module in encoder:
            x = module(x, edges)
        return x
    
    def handle_encoder_output(self, x):
        x = self.pooling(x)
        x = rearrange(x, 'b s n d -> b s (n d)')
        assert x.size()[-1] == self.d_model
        return x
    
    
    
    def forward(self, lhand, rhand, pose, hand_edges, pose_edges):
        """
        :param x: [b, s, n, xy]
        :param edges: [2, num_edges]
        """
        lhand = self.encoder_forward(lhand, hand_edges, self.encoder_lhand)
        rhand = self.encoder_forward(rhand, hand_edges, self.encoder_rhand)
        pose = self.encoder_forward(pose, pose_edges, self.encoder_pose)
        # [b, s, n, d_model]
        
        assumble = [lhand, rhand, pose]
        assumble = list(map(self.handle_encoder_output, assumble))
        assumble = torch.cat(assumble, dim=-1)
        
        x = rearrange(assumble, 'b s c -> b c s')
        x =self.decoder(x)
        x = rearrange(x, 'b c s -> b s c')
        return x