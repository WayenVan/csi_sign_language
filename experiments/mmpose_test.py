import sys 
import os
from pathlib import Path

sys.path.append('src')

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmengine.dataset import Compose, pseudo_collate
from csi_sign_language.utils import print_children
from csi_sign_language.dataset.phoenix14 import Phoenix14SegDatset
from einops import rearrange

phoenix_dir = os.path.join(Path(__file__).resolve().parent, '../dataset/phoenix2014-release')
register_all_modules()

config_file = 'mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'

dataset = Phoenix14SegDatset(phoenix_dir, length_time=320, length_glosses=40, padding_mode='back')
# please prepare an image with person

a = dataset[0][0][:, :, :, ::-1]

pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)


print(pipeline(dict(img=a[0])))
# for im in a:
#     b = pipeline(dict(img=im))
#     results = inference_topdown(model, im)
# print(results)