'''
Reference to MobileNetV2 implementation:
    - https://github.com/ShowLo/MobileNetV2/blob/master/mobileNetV2.py
    - https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import solver.solver as solver
import solver.solver_v2 as solver_v2
import solver.solver_v3 as solver_v3

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    temp = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if temp < 0.9 * v:
        temp += divisor
    return temp

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super(Block, self).__init__()
        self.mask = (stride == 1 and in_channels == out_channels)
        
        channels = expansion * in_channels
        
        if expansion == 1:
            self.layers = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        self.shortcut = nn.Sequential()
            
    def forward(self, x):
        if self.mask:
            return self.layers(x) + self.shortcut(x)
        return self.layers(x)
    
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=100, input_size=224, width_multiplier=1.0, init_weight=True, diff_drop=True):
        super(MobileNetV2, self).__init__()
        
        self.cfg = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.diff_drop = diff_drop
        
        input_channels = _make_divisible(32 * width_multiplier, 4 if width_multiplier == 0.1 else 8)
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, input_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True),
        )
        
        layers = []
        for t, c, n, s in self.cfg:
            output_channels = _make_divisible(c * width_multiplier, 4 if width_multiplier == 0.1 else 8)
            for i in range(n):
                layers.append(Block(input_channels, output_channels, s if i == 0 else 1, t))
                input_channels = output_channels
        self.features = nn.Sequential(*layers)
        
        output_channels = _make_divisible(1280 * width_multiplier, 4 if width_multiplier == 0.1 else 8) if width_multiplier > 1.0 else 1280
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True),
        )
        
        self.differential_dropout = None
        if self.diff_drop == "v1":
            self.differential_dropout = solver.DifferentialDropout()
        elif self.diff_drop == "v2":
            self.differential_dropout = solver_v2.DifferentialDropout()
        elif self.diff_drop == "v3":
            self.differential_dropout = solver_v3.DifferentialDropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(output_channels, num_classes)
        
        if init_weight:
            self._init_weights()
            
    def forward(self, x, epoch=None):
        x = self.conv_first(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        if self.training:
            if self.diff_drop == "v3":
                x = self.differential_dropout(x, epoch)
            elif self.diff_drop == "v1" or self.diff_drop == "v2":
                x = self.differential_dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            
def mobilenet_v2(**kwargs):
    return MobileNetV2(**kwargs)