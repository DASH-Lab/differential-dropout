import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np

class DifferentialDropout(nn.Module):
    def __init__(self, inplace=False):
        super(DifferentialDropout, self).__init__()
    
    def forward(self, x, module):
        if self.training:
            length = x.size(dim=0)
            temp = torch.reshape(x, (length, -1))
            
            corr_coef = torch.corrcoef(temp)
            
            temp = (temp - torch.mean(temp)) / torch.std(temp)
            minimum, maximum = torch.min(temp), torch.max(temp)
            bins = int(maximum - minimum) * 100
            
            batch_probs, _ = torch.histogram(torch.reshape(temp, (-1)), bins=bins, range=(minimum, maximum), density=True)
            batch_entropy = torch.sum(batch_probs * (0. - torch.log2(batch_probs)))
            local_entropy = torch.zeros(length)
            for i in range(length):
                factor1 = torch.mean(torch.abs(corr_coef[i]))
                probs, _ = torch.histogram(temp[i], bins=bins, range=(minimum, maximum), density=True)
                local_entropy[i] = torch.sum(probs * (0. - torch.log2(probs)))
                factor2 = local_entropy[i] / batch_entropy
                if factor2 == 0:
                    factor1 = 0.0
                    factor2 = 1.0
                elif factor2 < 1.0:
                    factor2 = 1.0 / factor2
                
                p = 1.0 - factor1 / factor2
                mask = (torch.rand(x[i].shape) > p).float()
                x[i] = mask * x[i] / (1.0 - p)
                
            self.PseudoPruning(module=module, batch_entropy=batch_entropy, local_entropies=local_entropy)
        return x

    def PseudoPruning(module, batch_entropy, local_entropies):
        minimum, maximum = torch.min(local_entropies), torch.max(local_entropies)
        minimum = minimum if minimum < batch_entropy else batch_entropy
        maximum = maximum if maximum > batch_entropy else batch_entropy
        batch_entropy = (batch_entropy - minimum) / (maximum - minimum)
        local_entropies = (local_entropies - minimum) / (maximum - minimum)
        
        p = torch.sqrt(torch.mean(torch.square(local_entropies - batch_entropy)))
        
        for param in module.parameters():
            param.requires_grad = True
        
        for param in module.parameters():
            param.requires_grad = np.random.uniform() > p