import torch
import torchvision.models as models
import torch.nn as nn
import transformers as tf
import copy
from einops import rearrange
import torch_geometric.nn as gnn

class GCNBert(nn.Module):
    
    def __init__(self, d_model, in_channel, out_channel, num_node, bert: tf.BertModel) -> None:
        super().__init__()
        self.num_node = num_node
        for name, module in bert.named_children():
            if name == 'encoder':
                self.encoder = copy.deepcopy(module)
        self.BERT_EMBEDDING_DIM = self.get_bert_embedding_dim(self.encoder)
        assert num_node <= self.BERT_EMBEDDING_DIM, 'number of nodes must smaller than the bert embedding dim'
        self.gcn = gnn.GraphConv(in_channel, d_model)
        self.g2b_projection = nn.Linear(num_node*d_model, self.BERT_EMBEDDING_DIM)
        self.linear = nn.Linear(self.BERT_EMBEDDING_DIM, num_node*out_channel)
        

    def get_bert_embedding_dim(self, encoder):
        for name, module in encoder.named_children():
            if name == 'query':
                return module.in_features
            else:
                return self.get_bert_embedding_dim(module)
            
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
        x = self.g2b_projection(x)
        x = self.encoder(x).last_hidden_state
        x = self.linear(x)
        x = rearrange(x, 'b t (n out) -> b t n out', n=self.num_node)
        return x
    
    @staticmethod
    def create_model(d_model, in_channel, out_channel, num_node):
        """
        create model from pytorch hub
        """
        bert = tf.BertModel.from_pretrained('bert-base-uncased')
        return GCNBert(d_model, in_channel, out_channel, num_node, bert)
        

class GCNBERTV2(nn.Module):
    
    def __init__(self, d_model, in_channel, out_channel, num_node, bert: tf.BertModel) -> None:
        super().__init__()
        self.num_node = num_node
        for name, module in bert.named_children():
            if name == 'encoder':
                self.encoder = copy.deepcopy(module)
        self.BERT_EMBEDDING_DIM = self.get_bert_embedding_dim(self.encoder)
        assert num_node <= self.BERT_EMBEDDING_DIM, 'number of nodes must smaller than the bert embedding dim'
        self.gcn = gnn.GraphConv(in_channel, d_model)
        self.g2b_projection = nn.Linear(num_node*d_model, self.BERT_EMBEDDING_DIM)
        self.linear = nn.Linear(self.BERT_EMBEDDING_DIM, num_node*out_channel)
        

    def get_bert_embedding_dim(self, encoder):
        for name, module in encoder.named_children():
            if name == 'query':
                return module.in_features
            else:
                return self.get_bert_embedding_dim(module)
            
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
        x = rearrange(x, '(b t) n f -> t b (n f)', b=batch)
        
        #padding zeros in the last dimension, No
        # padding = torch.zeros(size=(batch, timporal, self.BERT_EMBEDDING_DIM-x.size()[-1]))
        # x = torch.cat((x, padding), dim=-1)

        #using bert
        x = self.g2b_projection(x)
        x = self.encoder(x).last_hidden_state
        x = self.linear(x)
        x = rearrange(x, 't b (n out) -> b t n out', n=self.num_node)
        return x
    
    @staticmethod
    def create_model(d_model, in_channel, out_channel, num_node):
        """
        create model from pytorch hub
        """
        bert = tf.BertModel.from_pretrained('bert-base-uncased')
        return GCNBert(d_model, in_channel, out_channel, num_node, bert)