import torch
import torch.nn as nn
from einops import rearrange

from .unet import Unet1d
from .graph import GCNBert

class GNNUnet(nn.Module):

    def __init__(self, d_model, in_channel, n_encoder, n_classes, num_node) -> None:
        super().__init__()
        self.gb = GCNBert.create_model(d_model, in_channel, d_model, num_node)
        self.encoders = nn.ModuleList([GCNBert.create_model(d_model, d_model, d_model, num_node) for i in range(n_encoder)])
        self.decoder = Unet1d(d_model * num_node, n_classes)
        
        
    def forward(self, x_, edges):
        """
        :param x: [b, s, n, xy]
        """

        x = self.gb(x_, edges)
        for module in self.encoders:
            x = module(x, edges)
        # [b, s, n, d_model]
        

        x = rearrange(x, 'b s n d -> b (n d) s')
        return self.decoder(x)