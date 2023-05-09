import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DifferentialDropout(nn.Module):
    def __init__(self, inplace=False):
        super(DifferentialDropout, self).__init__()

    def forward(self, x, epoch):
        if self.training:
            length = x.size(dim=0)
            mask = torch.zeros_like(x)
            temp = torch.reshape(x, (length, -1))
            
            corr_coef = torch.corrcoef(temp)
            temp_mean = torch.zeros_like(temp[0])
            for i in range(length):
                temp_mean += temp[i]
            temp_mean /= length
            total_mse = 0.0
            for i in range(length):
                total_mse += torch.mean(torch.square(temp[i] - temp_mean))
            
            total_unique = torch.numel(torch.unique(torch.round(temp)))
            for i in range(length):
                factor1 = torch.mean(torch.abs(corr_coef[i]))
                
                factor2 = torch.mean(torch.square(temp[i] - temp_mean)) / total_mse
                
                factor3 = torch.numel(torch.unique(torch.round(temp[i]))) / total_unique
                
                factor4 = epoch + np.square(epoch * np.sin(epoch) * np.cos(epoch))
                p = 1 - ((factor1 + (1 - factor2) + (1 - factor3)) / (3 * factor4))

                mask[i] = (torch.rand(x[i].shape).to(x.device) > p).float()
            x = mask * x / (1.0 - p)
        return x
    
def PseudoPruning(module, input, epoch):
    length = input.size(dim=0)
    temp = torch.reshape(input, (length, -1))
    
    corr_coef = torch.corrcoef(temp)
    temp_mean = torch.zeros_like(temp[0])
    for i in range(length):
        temp_mean += temp[i]
    temp_mean /= length
    total_mse = 0.0
    for i in range(length):
        total_mse += torch.mean(torch.square(temp[i] - temp_mean))
    
    total_unique = torch.numel(torch.unique(torch.round(temp)))
    
    p = 0.0
    for i in range(length):
        factor1 = torch.mean(torch.abs(corr_coef[i]))
        
        factor2 = torch.mean(torch.square(temp[i] - temp_mean)) / total_mse
        
        factor3 = torch.numel(torch.unique(torch.round(temp[i]))) / total_unique
        
        factor4 = epoch + np.square(epoch * np.sin(epoch) * np.cos(epoch))
        
        candidate = 1 - ((factor1 + (1 - factor2) + (1 - factor3)) / (3 * factor4))

        if candidate > p:
            p = candidate

    for param in module.parameters():
        mask = (torch.rand(param.grad.shape).to(param.grad.device) > p).float()
        param.grad = param.grad.mul(mask)