'''
Reference to ViT Implementation: 
    - https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
    - https://deep-learning-study.tistory.com/807
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import solver.solver as solver
import solver.solver_v2 as solver_v2
import solver.solver_v3 as solver_v3

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_size: int = 768, img_size = 224) -> None:
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embedding_size, patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, embedding_size))
        
    def forward(self, x):
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=x.shape[0])
        x = torch.cat([cls_tokens, x], dim=1) + self.positions
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int = 768, num_heads: int = 8, dropout: float = 0.) -> None:
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.keys = nn.Linear(embedding_size, embedding_size)
        self.queries = nn.Linear(embedding_size, embedding_size)
        self.values = nn.Linear(embedding_size, embedding_size)
        self.attention_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(embedding_size, embedding_size)
        
    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), 'b n (h d) -> b h n d', h=self.num_heads)
        keys = rearrange(self.keys(x), 'b n (h d) -> b h n d', h=self.num_heads)
        values = rearrange(self.values(x), 'b n (h d) -> b h n d', h=self.num_heads)
        
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        attention = F.softmax(energy, dim=-1) / (self.embedding_size ** (1/2))
        attention = self.attention_drop(attention)
        
        out = torch.einsum('bhal, bhlv -> bhav', attention, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        x = self.fn(x, **kwargs) + x
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, embedding_size: int, expansion: int = 4, drop_prob: float = 0.0):
        super(FeedForwardBlock, self).__init__(
            nn.Linear(embedding_size, expansion * embedding_size),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(expansion * embedding_size, embedding_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, embedding_size: int = 768, drop_prob: float = 0.0, forward_expansion: int = 4, forward_drop_prob: float = 0.0, ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embedding_size),
                MultiHeadAttention(embedding_size=embedding_size, **kwargs),
                nn.Dropout(drop_prob),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embedding_size),
                FeedForwardBlock(embedding_size=embedding_size, expansion=forward_expansion, drop_prob=forward_drop_prob),
                nn.Dropout(drop_prob),
            )),
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, ** kwargs):
        super(TransformerEncoder, self).__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Module):
    def __init__(self, embedding_size=768, num_classes=100, diff_drop="v0"):
        super(ClassificationHead, self).__init__()
        
        self.diff_drop = diff_drop
        
        self.reduce_norm = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(embedding_size),
        )
        
        self.differential_dropout = None
        if self.diff_drop == "v1":
            self.differential_dropout = solver.DifferentialDropout()
        elif self.diff_drop == "v2":
            self.differential_dropout = solver_v2.DifferentialDropout()
        elif self.diff_drop == "v3":
            self.differential_dropout = solver_v3.DifferentialDropout()
        
        self.fc = nn.Linear(embedding_size, num_classes)
    
    def forward(self, x, epoch=None):
        p = 0.0
        x = self.reduce_norm(x)
        if self.training:
            if self.diff_drop == "v3":
                x, p = self.differential_dropout(x, epoch)
            elif self.diff_drop == "v1" or self.diff_drop == "v2":
                x, p = self.differential_dropout(x)
        x = self.fc(x)
        
        return x, p

class ViT(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_size=768, img_size=224, depth=12, num_classes=100, diff_drop="v0",**kwargs):
        super(ViT, self).__init__()
        self.embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_size=embedding_size, img_size=img_size)
        self.encoder = TransformerEncoder(depth=depth, embedding_size=embedding_size, **kwargs)
        self.classification_head = ClassificationHead(embedding_size=embedding_size, num_classes=num_classes, diff_drop=diff_drop)
        
    def forward(self, x, epoch=None):
        x = self.embedding(x)
        x = self.encoder(x)
        x, p = self.classification_head(x, epoch)
        
        return x, p