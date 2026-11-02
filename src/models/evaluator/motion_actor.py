import torch
import torch.nn as nn
import numpy as np

from typing import List, Optional, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution
from mmengine.registry import MODELS
from mmengine.model import BaseModel

@MODELS.register_module()
class ActorAgnosticEncoder(nn.Module):
    def __init__(self, nfeats: int, vae: bool,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", max_len: int = -1, **kwargs) -> None:
        super().__init__()
        input_feats = nfeats
        self.skel_embedding = nn.Linear(input_feats, latent_dim)
        self.max_len = max_len
        self.vae = vae
        # Action agnostic: only one set of params
        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

    def lengths_to_mask(self, lengths, device):
        if self.max_len == -1:
            max_len = max(lengths)
            mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        else:
            mask = torch.arange(self.max_len, device=lengths.device).expand(len(lengths), self.max_len) < lengths.unsqueeze(1)
        return mask 

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths, device=device)
        mask = self.lengths_to_mask(lengths, device).to(device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        if self.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)

            # adding the distribution tokens for all sequences
            xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)

            # create a bigger mask, to allow attend to mu and logvar
            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

            # adding the embedding token for all sequences
            xseq = torch.cat((emb_token[None], x), 0)

            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        if self.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)
            dist = torch.distributions.Normal(mu, std)
            return dist
        else:
            return final[0]

@MODELS.register_module()
class ActorAgnosticDecoder(nn.Module):
    def __init__(self, nfeats: int,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1, max_len=-1, 
                 activation: str = "gelu", **kwargs) -> None:

        super().__init__()
        output_feats = nfeats
        self.max_len = max_len
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransDecoder = nn.TransformerDecoder(seq_trans_decoder_layer,
                                                     num_layers=num_layers)

        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z: Tensor, lengths: List[int]):
        mask = self.lengths_to_mask(lengths, z.device)
        latent_dim = z.shape[1]
        bs, nframes = mask.shape

        z = z[None]  # sequence of 1 element for the memory

        # Construct time queries
        time_queries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=time_queries, memory=z,
                                      tgt_key_padding_mask=~mask)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats
    
    def lengths_to_mask(self, lengths, device):
        if self.max_len == -1:
            max_len = max(lengths)
            mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        else:
            mask = torch.arange(self.max_len, device=lengths.device).expand(len(lengths), self.max_len) < lengths.unsqueeze(1)
        return mask 


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)