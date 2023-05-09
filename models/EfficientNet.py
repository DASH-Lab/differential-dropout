'''
Reference to EfficientNet Implementation: 
    - https://github.com/kuangliu/pytorch-cifar/blob/master/models/efficientnet.py
    - https://deep-learning-study.tistory.com/563
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import solver.solver as solver
import solver.solver_v2 as solver_v2
import solver.solver_v3 as solver_v3

class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return x * self.sigmoid(x)
    
class SE(nn.Module):
    def __init__(self, in_channels, r=4) -> None:
        super(SE, self).__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels * r),
            Swish(),
            nn.Linear(in_channels * r, in_channels),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x

class MBConv(nn.Module):
    expand = 6
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4) -> None:
        super(MBConv, self).__init__()
        
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * MBConv.expand, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
            Swish(),
            nn.Conv2d(in_channels * MBConv.expand, in_channels * MBConv.expand, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False, groups=in_channels * MBConv.expand),
            nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
            Swish(),
        )
        
        self.se = SE(in_channels * MBConv.expand, se_scale)
        
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * MBConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3),
        )
        
        self.shortcut = (stride == 1) and (in_channels == out_channels)
        
    def forward(self, x):
        temp = x
        residue = self.residual(x)
        x = residue * self.se(residue)
        x = self.project(x)
        
        if self.shortcut:
            x = temp + x
        
        return x

class SepConv(nn.Module):
    expand = 1
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4) -> None:
        super(SepConv, self).__init__()
            
        self.residue = nn.Sequential(
            nn.Conv2d(in_channels * SepConv.expand, in_channels * SepConv.expand, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False, groups=in_channels * SepConv.expand),
            nn.BatchNorm2d(in_channels * SepConv.expand, momentum=0.99, eps=1e-3),
            Swish(),
        )
        
        self.se = SE(in_channels * SepConv.expand, se_scale)
        
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * SepConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3),
        )
        
        self.shortcut = (stride == 1) and (in_channels == out_channels)
        
    def forward(self, x):
        temp = x
        residue = self.residue(x)
        x = residue * self.se(residue)
        x = self.project(x)
        
        if self.shortcut:
            x = temp + x
        
        return x
        
class EfficientNet(nn.Module):
    def __init__(self, num_classes=100, width=1.0, depth=1.0, scale=1.0, dropout=0.2, se_scale=4, init_weight=True, diff_drop=True) -> None:
        super(EfficientNet, self).__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_size = [3, 3, 5, 3, 5, 5, 3]
        depth = depth
        width = width
        
        channels = [int(x * width) for x in channels]
        repeats = [int(x * depth) for x in repeats]
            
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        self.diff_drop = diff_drop
        self.stages = []
        
        stage = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0], momentum=0.99, eps=1e-3),
        )
        self.stages.append(stage)
        
        for i in range(7):
            block = MBConv
            if i == 0:
                block = SepConv
            temp = self._make_Block(block, repeats[i], channels[i], channels[i + 1], kernel_size[i], strides[i], se_scale)
            self.stages.append(temp)
        
        stage = nn.Sequential(
            nn.Conv2d(channels[7], channels[8], 1, stride=1, bias=False),
            nn.BatchNorm2d(channels[8], momentum=0.99, eps=1e-3),
            Swish()
        )
        self.stages.append(stage)
        self.stages = nn.Sequential(*self.stages)
        
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        
        self.dropout = None
        self.differential_dropout = None
        if self.diff_drop == "v1":
            self.differential_dropout = solver.DifferentialDropout()
        elif self.diff_drop == "v2":
            self.differential_dropout = solver_v2.DifferentialDropout()
        elif self.diff_drop == "v3":
            self.differential_dropout = solver_v3.DifferentialDropout()
        else:
            self.dropout = nn.Dropout(p=dropout)
            
        self.fc = nn.Linear(channels[8], num_classes)
        
        if init_weight:
            self._init_weights()
    
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
    
    def _make_Block(self, block, repeats, in_channels, out_channels, kernel_size, stride, se_scale):
        strides = [stride] + [1] * (repeats - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, kernel_size, stride, se_scale))
            in_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def forward(self, x, epoch=None):
        x = self.upsample(x)
        x = self.stages(x)
        x = self.squeeze(x)
        if self.training:
            if self.diff_drop == "v3":
                x = self.differential_dropout(x, epoch)
            elif self.diff_drop == "v1" or self.diff_drop == "v2":
                x = self.differential_dropout(x)
            else:
                x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def efficientnet_b0(num_classes=100, diff_drop=True):
    return EfficientNet(num_classes=num_classes, width=1.0, depth=1.0, scale=224/224, dropout=0.2, se_scale=4, diff_drop=diff_drop)
def efficientnet_b1(num_classes=100, diff_drop=True):
    return EfficientNet(num_classes=num_classes, width=1.0, depth=1.1, scale=240/224, dropout=0.2, se_scale=4, diff_drop=diff_drop)
def efficientnet_b2(num_classes=100, diff_drop=True):
    return EfficientNet(num_classes=num_classes, width=1.1, depth=1.2, scale=260/224, dropout=0.3, se_scale=4, diff_drop=diff_drop)
def efficientnet_b3(num_classes=100, diff_drop=True):
    return EfficientNet(num_classes=num_classes, width=1.2, depth=1.4, scale=230/224, dropout=0.3, se_scale=4, diff_drop=diff_drop)
def efficientnet_b4(num_classes=100, diff_drop=True):
    return EfficientNet(num_classes=num_classes, width=1.4, depth=1.8, scale=380/224, dropout=0.4, se_scale=4, diff_drop=diff_drop)
def efficientnet_b5(num_classes=100, diff_drop=True):
    return EfficientNet(num_classes=num_classes, width=1.6, depth=2.2, scale=456/224, dropout=0.4, se_scale=4, diff_drop=diff_drop)
def efficientnet_b6(num_classes=100, diff_drop=True):
    return EfficientNet(num_classes=num_classes, width=1.8, depth=2.6, scale=528/224, dropout=0.5, se_scale=4, diff_drop=diff_drop)
def efficientnet_b7(num_classes=100, diff_drop=True):
    return EfficientNet(num_classes=num_classes, width=2.0, depth=3.1, scale=600/224, dropout=0.5, se_scale=4, diff_drop=diff_drop)