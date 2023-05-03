import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DifferentialDropout(nn.Module):
    def __init__(self, inplace=False):
        super(DifferentialDropout, self).__init__()
    
    def forward(self, x):
        if self.training:
            length = x.size(dim=0)
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
            p = 0.0
            for i in range(length):
                factor1 = torch.mean(torch.abs(corr_coef[i]))
                
                factor2 = torch.mean(torch.square(temp[i] - temp_mean)) / total_mse
                
                factor3 = torch.numel(torch.unique(torch.round(temp[i]))) / total_unique
                
                candidate = (1 - factor1) * factor2 * factor3

                if candidate > p:
                    p = candidate
            
            x = F.dropout(x, p=p.item(), training=True)
        return x

def PseudoPruning(module, input):
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
        
        candidate = ((1 - factor1) * factor2 * factor3)

        if candidate > p:
            p = candidate

    for param in module.parameters():
        param.grad = param.grad.mul((torch.empty(param.grad.size()).uniform_(0, 1)).to(param.grad.device) > p)