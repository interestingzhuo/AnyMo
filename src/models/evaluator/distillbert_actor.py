import torch, os

from typing import List, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution
from transformers import AutoTokenizer, AutoModel
from typing import Tuple
import numpy as np
from mmengine.registry import MODELS
from mmengine.model import BaseModel

@MODELS.register_module()
class DistilbertEncoderBase(BaseModel):
    def __init__(self, modelpath: str,
                 finetune: bool = False) -> None:
        super().__init__()

        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath)
        # Don't train the model
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size

    def get_last_hidden_state(self, texts: List[str],
                              return_mask: bool = False
                              ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(self.text_model.device))
        if not return_mask:
            return output.last_hidden_state
        return output.last_hidden_state, encoded_inputs.attention_mask.to(dtype=bool)

@MODELS.register_module()
class DistilbertActorAgnosticEncoder(DistilbertEncoderBase):
    def __init__(self, modelpath: str,
                 finetune: bool = False,
                 vae: bool = True,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__(modelpath=modelpath, finetune=finetune)

        encoded_dim = self.text_encoded_dim
        # Projection of the text-outputs into the latent space
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(encoded_dim, latent_dim))

        # TransformerVAE adapted from ACTOR
        # Action agnostic: only one set of params
        self.vae = vae
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

    def forward(self, texts: List[str]) -> Union[Tensor, Distribution]:
        text_encoded, mask = self.get_last_hidden_state(texts, return_mask=True)

        x = self.projection(text_encoded)
        bs, nframes, _ = x.shape
        # bs, nframes, totjoints, nfeats = x.shape
        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

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
            try:
                dist = torch.distributions.Normal(mu, std)
            except ValueError:
                pass
            return dist
        else:
            return final[0]


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
