import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.norm import get_norm

class transpose_block(nn.Module):
    
    def __init__(self, in_channels, output_channels, padding=1, stride=2, norm='batch'):
        super(transpose_block, self).__init__()
        
        self.conv = nn.ConvTranspose2d(in_channels, output_channels, 4, stride=stride, padding=padding, bias=False)
        if type(norm) == str:
            self.norm = get_norm(norm)(output_channels)
        else:
            self.norm = norm
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm != False:
            x = F.relu(self.norm(x))
        return x