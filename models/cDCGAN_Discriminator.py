import torch
import torch.nn as nn
from layers.conv_relu import conv_relu


class CDiscriminator(nn.Module):
    
    def __init__(self, in_channels, d=128, num_classes=10):
        super(CDiscriminator, self).__init__()
        
        self.conv1_cond = conv_relu(num_classes, d // 2)
        self.conv1 = conv_relu(in_channels, d // 2)
        self.conv2 = conv_relu(d, d * 2)
        self.conv3 = conv_relu(d * 2, d * 4)
        self.conv4 = conv_relu(d * 4, 1, stride=1, padding=0, norm=False)
        
    def forward(self, x, labels):
        labels = self.conv1_cond(labels)
        x = self.conv1(x)
        x = torch.cat([x, labels], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return F.sigmoid(x)