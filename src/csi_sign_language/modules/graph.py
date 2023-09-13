import torch
import torchvision.models as models
import torch.nn as nn
import transformers as tf
import copy
from einops import rearrange
import torch_geometric.nn as gnn
from .bert import *

class GCNBertLayer(nn.Module):
    """
    GCN + linear + bert + linear module
    """
    def __init__(self, d_model, in_channel, out_channel, num_node, bert_layer: tf.BertLayer) -> None:
        super().__init__()
        self.num_node = num_node
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.BERT_EMBEDDING_DIM = bert_layer.attention.self.query.in_features
        self.gcn = gnn.GraphConv(in_channel, d_model)
        self.bert_layer = BertLayerWrapper(d_model*num_node, out_channel*num_node, bert_layer)

    def forward(self, attributes, edge, mask=None):
        """
        :param attributes: [batch, timporal_sequence, num_nodes, xy]
        :param edge: [2, num_edges]
        :param mask: [batch, timporal_sequence]
        """
        batch = attributes.size()[0]
        timporal = attributes.size()[1]
        assert self.num_node == attributes.size()[-2]
        
        #using GCN
        x = rearrange(attributes, 'b t n f -> (b t) n f')
        x = self.gcn(x, edge_index=edge)
        x = rearrange(x, '(b t) n f -> b t (n f)', b=batch)
        #padding zeros in the last dimension, No
        # padding = torch.zeros(size=(batch, timporal, self.BERT_EMBEDDING_DIM-x.size()[-1]))
        # x = torch.cat((x, padding), dim=-1)

        #using bert
        x = self.bert_layer(x)
        x = rearrange(x, 't b (n out) -> b t n out', n=self.num_node)
        return x