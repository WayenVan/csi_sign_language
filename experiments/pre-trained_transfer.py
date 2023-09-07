import torch
import torchvision.models as models
import torch.nn as nn
import transformers as tf
import copy
from einops import rearrange
import torch_geometric.nn as gnn

import sys
sys.path.append('src')
from csi_sign_language.utils.logger import build_logger

logger = build_logger('test', 'experiments/log.log')
# model = models.resnet18(pretrained=True)
# Recursive function to print all layers in a nested module
def print_nested_children(module, prefix=''):
    for name, child in module.named_children():
        print(prefix + name + ':', child)
        if isinstance(child, nn.Module):
            print_nested_children(child, prefix=prefix + '  ')
# print_nested_children(model)

def print_children(module, prefix=''):
    for name, child in module.named_children():
        logger.info(prefix + name + ':' + str(child))

bert = tf.BertModel.from_pretrained('bert-base-uncased')
print_children(bert)