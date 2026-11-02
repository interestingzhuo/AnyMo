import torch, random
import torch.nn as nn
from random import randrange
from mmengine.registry import MODELS

from .fsq import FSQ

@MODELS.register_module()
class ResidualFSQShare(nn.Module):
    def __init__(self, levels, dim, 
                 num_quantizers=2, quantize_dropout_prob=0.2, **kwargs):
        super().__init__(**kwargs)
        self.num_quantizers = num_quantizers
        self.quantize_dropout_prob = quantize_dropout_prob
        self.quantizer = FSQ(levels=levels, dim=dim)
    
    def forward(self, x, **kwargs):
        B, C, T = x.shape

        residual = x
        quantized_out = 0.0
        all_losses, all_indices, all_perplexity, all_activate = [], [], [], []
        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob

        start_drop_quantize_index = self.num_quantizers
        if should_quantize_dropout:
            start_drop_quantize_index = randrange(0, self.num_quantizers) 
            null_indices = torch.full((B, T), -1., device=x.device, dtype=torch.long)
        quantizers = [self.quantizer for _ in range(self.num_quantizers)]
        
        for quantizer_index, quantizer in enumerate(quantizers):
            if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                all_indices.append(null_indices) # type: ignore
                continue
            quantized, commit_loss, perplexity, activate, code_idx = quantizer(residual)

            residual -= quantized.detach()
            quantized_out += quantized
            all_activate.append(activate)

            all_indices.append(code_idx.view(B, -1))
            all_losses.append(commit_loss)
            all_perplexity.append(perplexity)

        # stack all losses and indices
        all_activate = sum(all_activate) / len(all_activate)
        all_indices = torch.stack(all_indices, dim=-1)
        all_losses = sum(all_losses) / len(all_losses)
        all_perplexity = sum(all_perplexity) / len(all_perplexity)

        return quantized_out, all_losses, all_perplexity, all_activate, all_indices
