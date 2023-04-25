import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int=1) -> None:
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),   
        )
        
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()
        
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )
        
    def forward(self, x):
        temp = x
        x = self.layer(x) + self.shortcut(temp)
        x = self.relu(x)
        return x

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super(BottleNeck, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
    
    def forward(self, x):
        temp = x
        x = self.layer(x) + self.shortcut(temp)
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, init_weights=True) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.conv2 = self.make_layer(block, 64, num_blocks[0], 1)
        self.conv3 = self.make_layer(block, 128, num_blocks[1], 2)
        self.conv4 = self.make_layer(block, 256, num_blocks[2], 1)
        self.conv5 = self.make_layer(block, 512, num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        if init_weights:
            self._init_weights()
        
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class Model():
    def resnet18(self, num_classes=1000):
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    def resnet34(self, num_classes=1000):
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    def resnet50(self, num_classes=1000):
        return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)
    def resnet101(self, num_classes=1000):
        return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)
    def resnet152(self, num_classes=1000):
        return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)