import torch
import torch.nn as nn

def conv3x3(in_channel, out_channel, stride = 1):
    return nn.Conv2d(in_channel, out_channel, kernel_size = 3,
                     stride = stride, padding = 1, bias = False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module): # [2, 2, 2]
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_channel = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace = True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channel, blocks, stride = 1):
        downsample = None
        if (stride != 1) or (self.in_channel != out_channel):
            downsample = nn.Sequential(conv3x3(self.in_channel, out_channel, stride = stride),
                                       nn.BatchNorm2d(out_channel))
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))

        self.in_channel = out_channel

        for i in range(1, blocks):
            layers.append(block(out_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x): # [B, 3, 32, 32]
        out = self.conv(x) # [B, 16, 32, 32]
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out) # [B, 16, 32, 32]
        out = self.layer2(out) # [B, 32, 16, 16]
        out = self.layer3(out) # [B, 64, 8, 8]
        out = self.avg_pool(out) # [B, 64, 1, 1]
        out = out.view(out.size(0), -1) # [B, 64]
        out = self.fc(out) # [B, 10]

        return out