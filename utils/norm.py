import torch.nn as nn
import functools

def get_norm(norm):
    if norm == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        return False