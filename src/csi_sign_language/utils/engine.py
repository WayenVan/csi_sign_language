
import torch
from einops import rearrange

def graph_video_crop(data, clip):
    """cut a batch of videos into clips with the clip sequence

    :param data: [b sequence ....]
    :param clip: number of the clip length
    :return cropped data
    """
    data: torch.tensor = rearrange(data, 'b (tmp clip) n xy -> (b tmp) clip n xy', clip=clip)
    return data 