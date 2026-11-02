r"""TMR: Text-to-Motion Retrieval
Using Contrastive 3D Human Motion Synthesis
Find more information about the model on the following website:
https://mathis.petrovich.fr/tmr

Args:
    motion_encoder: a module to encode the input motion features in the latent space (required).
    text_encoder: a module to encode the text embeddings in the latent space (required).
    motion_decoder: a module to decode the latent vector into motion features (required).
    vae: a boolean to make the model probabilistic (required).
    fact: a scaling factor for sampling the VAE (optional).
    sample_mean: sample the mean vector instead of random sampling (optional).
    lmd: dictionary of losses weights (optional).
    lr: learninig rate for the optimizer (optional).
    temperature: temperature of the softmax in the contrastive loss (optional).
    threshold_selfsim: threshold used to filter wrong negatives for the contrastive loss (optional).
    threshold_selfsim_metrics: threshold used to filter wrong negatives for the metrics (optional).
"""
import torch, os
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from typing import Optional

@MODELS.register_module()
class TMR(BaseModel):
    def __init__(self, motion_encoder_cfg, text_encoder_cfg, motion_decoder_cfg, 
                 temperature: float = 0.7, threshold_selfsim: float = 0.80, **kwargs):
        super().__init__(**kwargs)
        self.motion_encoder = MODELS.build(motion_encoder_cfg)
        self.text_encoder = MODELS.build(text_encoder_cfg)
        self.motion_decoder = MODELS.build(motion_decoder_cfg)

        # use distilbert text embedding
        model_path = 'checkpoints/distilbert-base-uncased'
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_model = AutoModel.from_pretrained(model_path)
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False

        # losses
        self.reconstruction_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.latent_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.kl_loss_fn = KLLoss()

        # adding the contrastive loss
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=temperature, threshold_selfsim=threshold_selfsim)

    def encode_text(self, captions):
        assert isinstance(captions, list)
        with torch.no_grad():
            text_tokens = self.tokenizer(captions, return_tensors="pt", padding=True)
            output = self.text_model(**text_tokens.to(self.text_model.device))
        mask = text_tokens.attention_mask.to(dtype=bool)
        text_embeddings = output.last_hidden_state # B, max_length, C

        mask_expanded = mask.unsqueeze(-1).expand(text_embeddings.shape).float()
        sentence_embeddings = torch.sum(text_embeddings * mask_expanded, 1) / torch.clamp(
            mask_expanded.sum(1), min=1e-9)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        dists = self.text_encoder(text_embeddings, mask).unbind(1)
        mu, logvar = dists

        # Reparameterization trick
        std = logvar.exp().pow(0.5)
        eps = std.data.new(std.size()).normal_()
        latent_vectors = mu + eps * std
        return latent_vectors, dists, sentence_embeddings

    def encode_motion(self, motion: torch.Tensor, 
                      mask: Optional[torch.Tensor]=None, 
                      motion_length: Optional[torch.Tensor]=None):
        assert mask is not None or motion_length is not None
        if mask is None:
            max_len = motion_length.max() # type: ignore
            mask = torch.arange(max_len, device=motion.device).expand( # type: ignore
                motion_length.shape[0], max_len) < motion_length.unsqueeze(1) # type: ignore
        x = self.motion_encoder(motion, mask)
        dists = x.unbind(1)
        mu, logvar = dists

        # Reparameterization trick
        std = logvar.exp().pow(0.5)
        eps = std.data.new(std.size()).normal_()
        latent_vectors = mu + eps * std
        return latent_vectors, dists
    
    def forward(self, motion, motion_length, text_data, mode='loss', **kwargs): # type: ignore
        '''
        motion: Tensor [B, T, C]
        motion_length: Tensor [B, ]
        condition_data: List[str]
        '''
        # motion = torch.cat([motion, trans], dim=-1)
        # generate motion mask
        max_len = motion_length.max()
        mask = torch.arange(max_len, device=motion.device).expand(
            motion_length.shape[0], max_len) < motion_length.unsqueeze(1)
        m_latents, m_dists = self.encode_motion(motion, mask=mask)
        m_motions = self.motion_decoder(m_latents, mask)

        t_latents, t_dists, sent_emb = self.encode_text(text_data)
        t_motions = self.motion_decoder(t_latents, mask)

        if mode == 'loss':
            losses = dict()

            # Reconstructions losses
            losses["recons_loss"] = (
                + self.reconstruction_loss_fn(t_motions, motion) # text -> motion
                + self.reconstruction_loss_fn(m_motions, motion) # motion -> motion
            ) * 1.0

            # VAE losses
            # Create a centred normal distribution to compare with logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)

            losses["kl_loss"] = (
                self.kl_loss_fn(t_dists, m_dists)  # text_to_motion
                + self.kl_loss_fn(m_dists, t_dists)  # motion_to_text
                + self.kl_loss_fn(m_dists, ref_dists)  # motion
                + self.kl_loss_fn(t_dists, ref_dists)  # text
            ) * 1.0e-5

            # Latent manifold loss
            losses["latent_loss"] = self.latent_loss_fn(t_latents, m_latents) * 1.0e-5
            # TMR: adding the contrastive loss
            losses["contrastive_loss"] = self.contrastive_loss_fn(t_latents, m_latents, sent_emb) * 0.1
            return losses
        else:
            return t_latents, m_latents, sent_emb

# For reference
# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
class KLLoss:
    def __call__(self, q, p):
        mu_q, logvar_q = q
        mu_p, logvar_p = p

        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()


class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim

    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    def __call__(self, x, y, sent_emb=None):
        
        bs, device = len(x), x.device
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature

        if sent_emb is not None and self.threshold_selfsim:
            # put the threshold value between -1 and 1
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Filtering too close values
            # mask them by putting -inf in the sim_matrix
            selfsim = sent_emb @ sent_emb.T
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            sim_matrix[idx] = -torch.inf
            # sim_matrix[idx] = -1e2

        labels = torch.arange(bs, device=device)
        
        total_loss = (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        # print(total_loss)
        return total_loss
