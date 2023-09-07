"""
Modify the pre-trained bert structure to fit other function
"""
from torch import nn
from transformers.models.bert import BertLayer

class BertLayerWrapper(nn.Module):
    
    def __init__(self, in_features, out_features, bertlayer: BertLayer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.BERT_FEATURE_DIM = bertlayer.attention.self.query.in_features
        self.in_linear = nn.Linear(in_features, self.BERT_FEATURE_DIM)
        self.out_linear = nn.Sequential(
            nn.Linear(self.BERT_FEATURE_DIM, out_features),
            nn.GELU()
        )
        self.bertlayer = bertlayer
    
    def forward(self, x):
        """
        :param x: [s b d]
        """
        x = self.in_linear(x)
        x = self.bertlayer(x)
        x = self.out_linear(x)
        return x