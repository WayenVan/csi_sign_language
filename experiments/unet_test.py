import torch
import sys
sys.path.append('src')
from csi_sign_language.utils import print_children, print_nested_children
from csi_sign_language.model.unet import Unet1d

net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)

x = torch.ones(10, 768, 32)
net = Unet1d(768, 100)
print(net(x).shape)
