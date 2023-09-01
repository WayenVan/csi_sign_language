import torch
import torch.nn as nn

def print_nested_children(module, prefix=''):
    for name, child in module.named_children():
        print(prefix + name + ':', child)
        if isinstance(child, nn.Module):
            print_nested_children(child, prefix=prefix + '  ')

def print_children(module, prefix=''):
    for name, child in module.named_children():
        print(prefix + name + ':', child)
