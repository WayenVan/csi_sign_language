import torch
import torch.nn as nn
import logging
def print_nested_children(module, logger: logging.Logger, prefix=''):
    for name, child in module.named_children():
        logger.info(f'{prefix}{name}:{child}')
        if isinstance(child, nn.Module):
            print_nested_children(child, logger, prefix=prefix + '  ')

def print_children(module, logger: logging.Logger, prefix=''):
    for name, child in module.named_children():
        logger.info(prefix + name + ':' + str(child))


def print_gradient(model, logger: logging.Logger):
    for name, param in model.named_parameters():
        if param.grad is not None:
            logger.info(f'Layer: {name}, Gradient Shape: {param.grad.shape}')
            logger.info(param.grad)