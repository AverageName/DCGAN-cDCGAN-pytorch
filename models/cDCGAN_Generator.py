import torch
import torch.nn as nn
from layers.transpose_block import transpose_block


class CGenerator(nn.Module):
    
    def __init__(self, in_channels, d=128, num_classes=10):
        super(CGenerator, self).__init__()
        
        self.conv1_cond = transpose_block(num_classes, d * 2, padding=0, stride=1)
        self.conv1 = transpose_block(in_channels, d *2, padding=0, stride=1)
        self.conv2 = transpose_block(d * 4, d * 2)
        self.conv3 = transpose_block(d * 2, d)
        self.conv4 = transpose_block(d, 1, norm=False)
        
    def forward(self, x, labels):
        labels = self.conv1_cond(labels)
        x = self.conv1(x)
        x = torch.cat([x, labels], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return F.tanh(x)