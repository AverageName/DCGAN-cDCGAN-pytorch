import torch
import torch.nn as nn
from utils.norm import get_norm

class conv_relu(nn.Module):
    
    def __init__(self, in_channels, output_channels, kernel_size=4, stride=2, padding=1, norm='batch'):
        super(conv_relu, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.lrelu = nn.LeakyReLU(0.2)
        if type(norm) == str:
            self.norm = get_norm(norm)(output_channels)
        else:
            self.norm = norm
            
    def forward(self, x):
        x = self.conv(x)
        if self.norm != False:
            x = self.lrelu(self.norm(x))
        return x