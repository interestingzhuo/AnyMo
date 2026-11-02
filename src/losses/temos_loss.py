import torch
import torch.nn as nn
from mmengine.registry import MODELS

import torch
import torch.nn.functional as F
import numpy as np

class InfoNCE:
    def __init__(self, t):
        # pass
        self.t = t

    def __call__(self, f, dist):
        '''
        f_motion: N x d 
        f_text: N x d 
        '''
        t = self.t
        f_motion, f_text = f[0], f[1]
        N, d = f_motion.shape[0], f_motion.shape[1]

        
        Emb_motion = F.normalize(f_motion, dim=1)
        Emb_text = F.normalize(f_text, dim=1)
        
        t = torch.tensor(t).to(f_motion.device)
        logits = torch.mm(Emb_motion, Emb_text.T)
        # logits = torch.mm(Emb_motion, Emb_text.T) / torch.exp(t)
        if dist is not None:
            text_logits = dist.detach()
            mask = torch.where(torch.logical_and(text_logits > 0.85, text_logits < 1.0-1e-100), torch.tensor(float('-inf')).to(f_motion.device), torch.tensor(1.0e100).to(f_motion.device))
            mask.diagonal().fill_(float('inf'))
            logits = torch.min(mask, logits)
            # mask = torch.where((torch.logical_and(text_logits > 0.985, text_logits < 1.0-1e-100)), torch.tensor(float('-inf')).cuda(), torch.tensor(1.0e100).cuda())
            # logits = torch.min(mask, logits)
        
        N = f_motion.shape[0]
        labels = torch.arange(N).to(f_motion.device)
        
        loss_m = F.cross_entropy(logits / t, labels)
        loss_t = F.cross_entropy(logits.T / t, labels)

        loss = (loss_m + loss_t) / 2
        
        return loss
        
    def __repr__(self):
        return "InfoNCE()"

@MODELS.register_module()
class TemosLosses(nn.Module):
    """
    Loss
    Modify loss
    refer to temos loss
    add loss like deep-motion-editing
    'gen_loss_total': l_total,
    'gen_loss_adv': l_adv,
    'gen_loss_recon_all': l_rec,
    'gen_loss_recon_r': l_r_rec,
    'gen_loss_recon_s': l_s_rec,
    'gen_loss_feature_all': l_ft,
    'gen_loss_feature_r': l_ft_r,
    'gen_loss_feature_s': l_ft_s,
    'gen_loss_feature_t': l_ft_t,
    'gen_loss_quaternion': l_qt,
    'gen_loss_twist': l_tw,
    'gen_loss_triplet': l_triplet,
    'gen_loss_joint': l_joint,
           
    """

    def __init__(self, 
                 vae=True, 
                 mode='xyz', 
                 use_infonce=True, 
                 lambda_latent=1e-5, 
                 lambda_kl=1e-5, 
                 lambda_rec=1.0, 
                 lambda_gen=1.0, 
                 lambda_cycle=0.0, 
                 lambda_prior=0.0, 
                 lambda_infonce=0.1, 
                 infonce_temp=0.1, 
                 **kwargs):
        super().__init__()
        # Save parameters
        self.vae = vae
        self.mode = mode

        self.use_infonce = use_infonce

        if self.use_infonce:
            self.infonce_temp = infonce_temp

        loss_on_both = True
        force_loss_on_jfeats = True
        ablation_no_kl_combine = False
        ablation_no_kl_gaussian = False
        ablation_no_motionencoder = False

        infonce_use_latent = False

        self.loss_on_both = loss_on_both
        self.ablation_no_kl_combine = ablation_no_kl_combine
        self.ablation_no_kl_gaussian = ablation_no_kl_gaussian
        self.ablation_no_motionencoder = ablation_no_motionencoder

        self.infonce_use_latent = infonce_use_latent

        losses = []
        if mode == "xyz" or force_loss_on_jfeats:
            if not ablation_no_motionencoder:
                losses.append("recons_jfeats2jfeats")
            losses.append("recons_text2jfeats")
        if mode == "smpl":
            if not ablation_no_motionencoder:
                losses.append("recons_rfeats2rfeats")
            losses.append("recons_text2rfeats")
        else:
            ValueError("This mode is not recognized.")

        if vae or loss_on_both:
            kl_losses = []
            if not ablation_no_kl_combine and not ablation_no_motionencoder:
                kl_losses.extend(["kl_text2motion", "kl_motion2text"])
            if not ablation_no_kl_gaussian:
                if ablation_no_motionencoder:
                    kl_losses.extend(["kl_text"])
                else:
                    kl_losses.extend(["kl_text", "kl_motion"])
            losses.extend(kl_losses)

        if not self.vae or loss_on_both:
            if not ablation_no_motionencoder:
                losses.append("latent_manifold")
        losses.append("total")

        if self.use_infonce:
            losses.append("contrastive_infonce")


        for loss in losses:
            self.register_buffer(loss, torch.tensor(0.0))
        self.register_buffer("count", torch.tensor(0))
        self.losses = losses
        self._losses_func = {}
        self._params = {}

        for loss in losses:
            if loss != 'total':
                if loss.split('_')[0] == 'kl':
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = lambda_kl
                elif loss.split('_')[0] == 'recons':
                    self._losses_func[loss] = torch.nn.SmoothL1Loss(
                        reduction='mean')
                    self._params[loss] = lambda_rec
                elif loss.split('_')[0] == 'latent':
                    self._losses_func[loss] = torch.nn.SmoothL1Loss(
                        reduction='mean')
                    self._params[loss] = lambda_latent
                elif loss.split('_')[0] == 'cycle':
                    self._losses_func[loss] = torch.nn.SmoothL1Loss(
                        reduction='mean')
                    self._params[loss] = lambda_cycle
                elif loss.split('_')[0] == 'contrastive':
                    self._losses_func[loss] = InfoNCE(self.infonce_temp)
                    self._params[loss] = lambda_infonce
                else:
                    ValueError("This loss is not recognized.")

    def forward(self,
               f_text=None,
               f_motion=None,
               f_ref=None,
               lat_text=None,
               lat_motion=None,
               dis_text=None,
               dis_motion=None,
               dis_ref=None, 
               emb_dist=None):

        loss_dict = dict()
        if self.mode == "xyz" or self.force_loss_on_jfeats:
            if not self.ablation_no_motionencoder:
                loss_dict['recons_jfeats2jfeats_loss'] = self.get_loss("recons_jfeats2jfeats", f_motion, f_ref)
            loss_dict['recons_text2jfeats_loss'] = self.get_loss("recons_text2jfeats", f_text, f_ref)

        if self.mode == "smpl":
            raise NotImplementedError

        if self.vae or self.loss_on_both:
            if not self.ablation_no_kl_combine and not self.ablation_no_motionencoder:
                loss_dict['kl_text2motion_loss'] = self.get_loss("kl_text2motion", dis_text, dis_motion)
                loss_dict['kl_motion2text_loss'] = self.get_loss("kl_motion2text", dis_motion, dis_text)
            if not self.ablation_no_kl_gaussian:
                loss_dict['kl_text_loss'] = self.get_loss("kl_text", dis_text, dis_ref)
                if not self.ablation_no_motionencoder:
                    loss_dict['kl_motion_loss'] = self.get_loss("kl_motion", dis_motion, dis_ref)

        if not self.vae or self.loss_on_both:
            if not self.ablation_no_motionencoder:
                loss_dict['latent_manifold_loss'] = self.get_loss("latent_manifold", lat_text, lat_motion)
        if self.use_infonce:
            if self.infonce_use_latent:
                loss_dict['contrastive_infonce_loss'] = self.get_loss("contrastive_infonce", (lat_text, lat_motion), emb_dist)
            else:
                loss_dict['contrastive_infonce_loss'] = self.get_loss("contrastive_infonce", (dis_motion.loc, dis_text.loc), emb_dist)

        return loss_dict
    
    def get_loss(self, loss: str, outputs, inputs):
        val = self._losses_func[loss](outputs, inputs)
        weighted_loss = self._params[loss] * val
        return weighted_loss

class KLLoss:

    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti:

    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p) for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"
