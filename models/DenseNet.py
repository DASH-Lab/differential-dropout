'''
Reference to DenseNet Implementation: 
    - https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py
    - https://deep-learning-study.tistory.com/545
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleNeck, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 4 * growth_rate, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(),
            nn.Conv2d(4 * growth_rate, growth_rate, 3, stride=1, padding=1, bias=False),
        )
        
        self.shortcut = nn.Sequential()
        
    def forward(self, x):
        x = torch.cat([self.shortcut(x), self.layer(x)], 1)
        return x
    
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2, stride=2),
        )
        
    def forward(self, x):
        return self.layer(x)
        
class DenseNet(nn.Module):
    def __init__(self, block, num_blocks, growth_rate=12, reduction=0.5, num_classes=10, init_weights=True):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        
        num_channels = 2 * growth_rate
        
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        
        self.dense_module1 = self._make_dense_layers(block, num_channels, num_blocks[0])
        num_channels += num_blocks[0] * growth_rate
        out_channels = int(math.floor(num_channels * reduction))
        self.trans_module1 = Transition(num_channels, out_channels)
        num_channels = out_channels
        
        self.dense_module2 = self._make_dense_layers(block, num_channels, num_blocks[1])
        num_channels += num_blocks[1] * growth_rate
        out_channels = int(math.floor(num_channels * reduction))
        self.trans_module2 = Transition(num_channels, out_channels)
        num_channels = out_channels
        
        self.dense_module3 = self._make_dense_layers(block, num_channels, num_blocks[2])
        num_channels += num_blocks[2] * growth_rate
        out_channels = int(math.floor(num_channels * reduction))
        self.trans_module3 = Transition(num_channels, out_channels)
        num_channels = out_channels
        
        self.dense_module4 = self._make_dense_layers(block, num_channels, num_blocks[3])
        num_channels += num_blocks[3] * growth_rate
        
        self.batch_norm = nn.BatchNorm2d(num_channels)
        self.linear = nn.Linear(num_channels, num_classes)
        
        if init_weights:
            self._init_weights()
    
    def _make_dense_layers(self, block, in_channels, num_block):
        layers = []
        for i in range(num_block):
            layers.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dense_module1(x)
        x = self.trans_module1(x)
        x = self.dense_module2(x)
        x = self.trans_module2(x)
        x = self.dense_module3(x)
        x = self.trans_module3(x)
        x = self.dense_module4(x)
        x = F.avg_pool2d(F.relu(self.batch_norm(x)), 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def DenseNet121(num_classes=1000):
    return DenseNet(BottleNeck, [6, 12, 24, 16], growth_rate=32, num_classes=num_classes)
def DenseNet161(num_classes=1000):
    return DenseNet(BottleNeck, [6, 12, 36, 24], growth_rate=32, num_classes=num_classes)
def DenseNet169(num_classes=1000):
    return DenseNet(BottleNeck, [6, 12, 32, 32], growth_rate=32, num_classes=num_classes)
def DenseNet201(num_classes=1000):
    return DenseNet(BottleNeck, [6, 12, 48, 32], growth_rate=32, num_classes=num_classes)
    
    