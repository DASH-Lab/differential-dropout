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

class DifferentialDropout_v2(nn.Module):
    def __init__(self, inplace=False):
        super(DifferentialDropout, self).__init__()

    def forward(self, x, module):
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
            
            _, unique_overall = torch.unique(torch.round(temp), return_counts=True)
            unique_overall /= torch.sum(unique_overall)
            batch_entropy = torch.sum(unique_overall * (0.0 - torch.log2(unique_overall)))
            for i in range(length):
                factor1 = torch.mean(torch.abs(corr_coef[i]))
                
                factor2 = torch.mean(torch.square(temp[i] - temp_mean)) / total_mse
            
                _, unique_local = torch.unique(torch.round(temp[i]), return_counts=True)
                unique_local /= torch.sum(unique_local)
                local_entropy = torch.sum(unique_local * (0.0 - torch.log2(unique_local)))
                factor3 = local_entropy / batch_entropy

                if factor3 > 1.0:
                    factor3 = 1.0 / factor3
                p = (1.0 - factor1) * factor2 * factor3

                mask[i] = (torch.rand(x[i].shape) > p).float()
            x = mask.to(x.device) * x / (1.0 - p)
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
        mask = (torch.rand(param.grad.shape).to(param.grad.device) > p).float()
        param.grad = param.grad.mul(mask)
        
def PseudoPruning_v2(module, input):
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
    
    _, unique_overall = torch.unique(torch.round(temp), return_counts=True)
    unique_overall /= torch.sum(unique_overall)
    batch_entropy = torch.sum(unique_overall * (0.0 - torch.log2(unique_overall)))
    p = 0.0
    for i in range(length):
        factor1 = torch.mean(torch.abs(corr_coef[i]))
        
        factor2 = torch.mean(torch.square(temp[i] - temp_mean)) / total_mse
        
        _, unique_local = torch.unique(torch.round(temp[i]), return_counts=True)
        unique_local /= torch.sum(unique_local)
        local_entropy = torch.sum(unique_local * (0.0 - torch.log2(unique_local)))
        factor3 = local_entropy / batch_entropy

        if factor3 > 1.0:
            factor3 = 1.0 / factor3
        
        candidate = ((1 - factor1) * factor2 * factor3)

        if candidate > p:
            p = candidate

    for param in module.parameters():
        mask = (torch.rand(param.grad.shape).to(param.grad.device) > p).float()
        param.grad = param.grad.mul(mask)