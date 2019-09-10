import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.transpose_block import transpose_block


class Generator(nn.Module):
    
    def __init__(self, in_channels, norm='batch', d=128):
        super(Generator, self).__init__()
        
        self.conv1 = transpose_block(in_channels, d * 4, padding=0, stride=1, norm=norm)
        self.conv2 = transpose_block(d * 4, d * 2, norm=norm)
        self.conv3 = transpose_block(d * 2, d, norm=norm)
        self.conv4 = transpose_block(d, 1, norm=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return F.tanh(x)