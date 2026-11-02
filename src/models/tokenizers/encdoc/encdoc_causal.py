import torch
import torch.nn as nn
from mmengine.registry import MODELS
from .resnet import CausalResnet1D, CausalConv1d

@MODELS.register_module()
class CausalEncoder(nn.Module):
    def __init__(self,
                 input_emb_width = 272,
                 hidden_size = 1024,
                 down_t = 2,
                 stride_t = 2,
                 width = 1024,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None,
                 latent_dim=16,
                 clip_range = []
                 ):
        super().__init__()
        self.clip_range = clip_range
        self.proj = nn.Linear(width, latent_dim*2)

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2      


        blocks.append(CausalConv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):   
            input_dim = width
            block = nn.Sequential(
                CausalConv1d(input_dim, width, filter_t, stride_t, 1),
                CausalResnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, hidden_size, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = self.model(x)
        x = x.transpose(1, 2)  
        x = self.proj(x)        
        mu, logvar = x.chunk(2, dim=2)             
        logvar = torch.clamp(logvar, self.clip_range[0], self.clip_range[1])
        z = self.reparameterize(mu, logvar) 

        return z, mu, logvar

@MODELS.register_module()
class CausalDecoder(nn.Module):
    def __init__(self,
                 input_emb_width = 272,
                 hidden_size = 1024,
                 down_t = 2,
                 stride_t = 2,
                 width = 1024,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None
                 ):
        super().__init__()
        blocks = []
        
        blocks.append(CausalConv1d(hidden_size, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                CausalResnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                CausalConv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(CausalConv1d(width, input_emb_width, 3, 1, 1))

        self.model = nn.Sequential(*blocks)

    def forward(self, z):
        z = z.transpose(1, 2)
        return self.model(z)

