import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.conv_relu import conv_relu


class Discriminator(nn.Module):
    
    def __init__(self, in_channels, norm='batch', d=128):
        super(Discriminator, self).__init__()
        
        self.conv1 = conv_relu(in_channels, d, norm=norm)
        self.conv2 = conv_relu(d, d * 2, norm=norm)
        self.conv3 = conv_relu(d * 2, d * 4, norm=norm)
        self.conv4 = conv_relu(d * 4, 1, stride=1, padding=0, norm=False)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return F.sigmoid(x)