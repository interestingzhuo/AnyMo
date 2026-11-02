import torch, os
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from typing import Optional, List
from torch import Tensor
from torch.distributions.distribution import Distribution
import torch.nn.functional as F

@MODELS.register_module()
class TEMOS(BaseModel):
    def __init__(self, vae, text_encoder_cfg, motion_encoder_cfg, motion_decoder_cfg, loss_cfg, **kwargs):
        super().__init__(**kwargs)

        self.is_vae = vae
        self.loss_cfg = loss_cfg

        self.losses = MODELS.build(loss_cfg)
        self.textencoder = MODELS.build(text_encoder_cfg)
        self.motionencoder = MODELS.build(motion_encoder_cfg)
        self.motiondecoder = MODELS.build(motion_decoder_cfg)
        self.fact = None
        if loss_cfg.use_infonce_filter:
            from sentence_transformers import SentenceTransformer
            self.filter_model = SentenceTransformer('checkpoints/sentence-transformers-paraphrase-MiniLM-L6-v2')
            for param in self.filter_model.parameters():
                param.requires_grad = False

    def sample_from_distribution(self, distribution: Distribution, *,
                                 fact: Optional[bool] = None,
                                 sample_mean: Optional[bool] = False) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return distribution.loc

        # Reparameterization trick
        if fact is None:
            return distribution.rsample()

        # Resclale the eps
        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector

    def text_to_motion_forward(self, text_sentences: List[str], lengths: List[int], *,
                               return_latent: bool = False):
        # Encode the text to the latent space
        if self.is_vae:
            distribution = self.textencoder(text_sentences)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector = self.textencoder(text_sentences)

        features = self.motiondecoder(latent_vector, lengths)

        if not return_latent:
            return features
        return features, latent_vector, distribution

    def motion_to_motion_forward(self, features,
                                 lengths: Optional[List[int]] = None,
                                 return_latent: bool = False):
        if self.is_vae:
            distribution = self.motionencoder(features, lengths)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector: Tensor = self.motionencoder(features, lengths)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        # datastruct = self.Datastruct(features=features)

        if not return_latent:
            return features
        return features, latent_vector, distribution
   
    def forward(self, motion, motion_length, text_data, mode='loss', **kwargs):
        emb_dist = None
        pad_m = torch.zeros((motion.shape[0], 300 - motion_length.max(), 272)).to(motion)
        motion = torch.cat([motion, pad_m], dim=1)

        if self.loss_cfg.use_infonce and self.loss_cfg.use_infonce_filter:
            with torch.no_grad():
                text_embedding = self.filter_model.encode(text_data)
                text_embedding = torch.tensor(text_embedding).to(motion)
                normalized = F.normalize(text_embedding, p=2, dim=1)
                emb_dist = normalized.matmul(normalized.T)

        # Encode the text/decode to a motion
        ret = self.text_to_motion_forward(text_data, motion_length, return_latent=True)
        feat_from_text, latent_from_text, distribution_from_text = ret

        # Encode the motion/decode to a motion
        ret = self.motion_to_motion_forward(motion, motion_length, return_latent=True)
        feat_from_motion, latent_from_motion, distribution_from_motion = ret

        # Compare to a Normal distribution
        if self.is_vae:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(distribution_from_text.loc)
            scale_ref = torch.ones_like(distribution_from_text.scale)
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            distribution_ref = None
        if mode == 'loss':
            loss = self.losses(f_text=feat_from_text, f_motion=feat_from_motion,
                            f_ref=motion, lat_text=latent_from_text,
                            lat_motion=latent_from_motion, dis_text=distribution_from_text,
                            dis_motion=distribution_from_motion, dis_ref=distribution_ref, 
                            emb_dist=emb_dist)
            return loss
        else:
            return distribution_from_text.loc, distribution_from_motion.loc, emb_dist
