import torch
import sys

sys.path.append('src')

from csi_sign_language.models.models import *
from csi_sign_language.utils.inspect import *
from csi_sign_language.utils.logger import build_logger

from torchinfo import summary

logger = build_logger('main', 'experiments/log.log')
model = ResnetTransformer('haha', 1024, 1, 100)
summary(model, input_size=(2, 16, 3, 256, 256))
model.eval()