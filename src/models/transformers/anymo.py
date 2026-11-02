import torch, math
import torch.nn.functional as F
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from typing import Dict, List
from torch.distributions.categorical import Categorical
import numpy as np
from random import random
from ...utils.motion import recover_from_local_position_fast
from ..tokenizers.encdoc.encdoc_attn import length_to_mask

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask

def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

def top_k(logits, thres = 0.9, dim = 1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim = dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    return probs

@MODELS.register_module()
class AnyMo(BaseModel):
    def __init__(
        self, 
        vqvae_cfg: Dict,
        transformer_cfg: Dict,

        max_motion_len: int,
        motion_downsample_ratio: int, 
        audio_downsample_ratio: int, 

        text_encoder_cfg: Dict=None,
        traj_encoder_cfg: Dict=None, 
        audio_encoder_cfg: Dict=None,
        
        stage=1, 
        inference_timesteps=6, 
        cond_types = ['text'], 
        **kwargs
    ):
        super().__init__(**kwargs)

        self.transformer = MODELS.build(transformer_cfg)
        if text_encoder_cfg is not None:
            self.text_encoder = MODELS.build(text_encoder_cfg)
        if traj_encoder_cfg is not None:
            self.traj_encoder = MODELS.build(traj_encoder_cfg)
        if audio_encoder_cfg is not None:
            self.audio_encoder = MODELS.build(audio_encoder_cfg)
        
        # 第一阶段tm训练，第二阶段只训练projector，第三阶段全量训练, 4阶段测评
        assert stage in [1, 2, 3, 4]
        self.stage = stage

        # build vqvae
        self.vqvae = MODELS.build(vqvae_cfg)
        self.vqvae.init_weights()
        for param in self.vqvae.parameters():
            param.requires_grad = False
        self.vqvae.eval()

        self.vocab_size = transformer_cfg.vocab_size # type: ignore
        self.n_embd = transformer_cfg.n_embd
        self.mask_idx = self.vocab_size - 1
        self.pad_idx = self.vocab_size - 2
        self.ignore_index = -100

        self.max_motion_len = max_motion_len
        self.motion_downsample_ratio = motion_downsample_ratio
        self.audio_downsample_ratio = audio_downsample_ratio
        self.topk_filter_thres = 0.9
        self.noise_schedule = cosine_schedule
        self.n_quant = len(self.vqvae.quantizer.quantizers)
        self.inference_timesteps = inference_timesteps
        self.cond_types = cond_types
        for cond_type in cond_types:
            assert cond_type in ['text', 'audio', 'traj']

        if self.stage == 2:
            self.init_weights()
            # only train audio and traj projector
            for param in self.transformer.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward_stage_1(self, motion, motion_length, text_data, mode='loss'):
        '''
        第一阶段只训t2m
        '''
        B = motion.shape[0]
        # 1. prepare text feature
        # text feat: B, text_token_cnt, C
        # text mask: B, text_token_cnt
        text_feat, text_mask = self.text_encoder(text_data)
        audio_feat = torch.zeros(B, 0, self.n_embd).to(text_feat)
        audio_mask = torch.zeros(B, 0).to(text_feat)
        traj_feat = torch.zeros(B, 0, self.n_embd).to(text_feat)
        traj_mask = torch.zeros(B, 0).to(text_feat)

        if mode == 'loss':
            tokens, label_tokens, non_pad_mask = self.prepare_train_input(motion, motion_length)
            loss =  self.forward_loss(tokens, label_tokens, text_feat, text_mask, audio_feat, audio_mask, traj_feat, traj_mask, non_pad_mask)
            return loss
        else:
            non_pad_mask = self.prepare_test_input(motion_length)
            return self.forward_predict(text_feat, text_mask, audio_feat, audio_mask, traj_feat, traj_mask, non_pad_mask)

    def forward_stage_2(self, motion, motion_length, audio_data, traj_data, audio_length, mode='loss', **kwargs):
        B = motion.shape[0]
        # 2. prepare audio and traj feature
        # audio feat: B, audio_token_cnt, C
        # traj feat: B, traj_token_cnt, C

        prob = random()
        if prob < 0.7: # 百分之70的语音数据
            audio_feat = self.audio_encoder(audio_data.permute(0, 2, 1)).permute(0, 2, 1)
            audio_mask = length_to_mask(audio_length // self.audio_downsample_ratio)
            traj_feat = torch.zeros(B, 0, self.n_embd).to(audio_feat)
            traj_mask = torch.zeros(B, 0).to(audio_feat)
            dtype = audio_feat.dtype
            device = audio_feat.device
        elif prob >= 0.7 and prob < 0.9: # 百分之20的轨迹数据
            traj_feat = self.traj_encoder(traj_data.permute(0, 2, 1)).permute(0, 2, 1)
            traj_mask = length_to_mask(motion_length // self.motion_downsample_ratio)
            audio_feat = torch.zeros(B, 0, self.n_embd).to(traj_feat)
            audio_mask = torch.zeros(B, 0).to(traj_feat)
            dtype = traj_feat.dtype
            device = traj_feat.device
        else:
            audio_feat = self.audio_encoder(audio_data.permute(0, 2, 1)).permute(0, 2, 1)
            audio_mask = length_to_mask(audio_length // self.audio_downsample_ratio)
            traj_feat = self.traj_encoder(traj_data.permute(0, 2, 1)).permute(0, 2, 1)
            traj_mask = length_to_mask(motion_length // self.motion_downsample_ratio)
            dtype = audio_feat.dtype
            device = audio_feat.device

        text_feat = torch.zeros(B, 0, self.n_embd).to(dtype).to(device)
        text_mask = torch.zeros(B, 0).to(dtype).to(device)

        if mode == 'loss':
            tokens, label_tokens, non_pad_mask = self.prepare_train_input(motion, motion_length)
            loss =  self.forward_loss(
                tokens, label_tokens, text_feat, text_mask, audio_feat, 
                audio_mask, traj_feat, traj_mask, non_pad_mask, motion=motion, mean=kwargs['mean'], 
                std=kwargs['std']
            )
            return loss
        else:
            non_pad_mask = self.prepare_test_input(motion_length)
            return self.forward_predict(text_feat, text_mask, audio_feat, audio_mask, traj_feat, traj_mask, non_pad_mask, **kwargs)
    
    def forward_stage_3(self, motion, motion_length, text_data, audio_data, traj_data, audio_length, mode='loss', **kwargs):
        B = motion.shape[0]
        has_traj = random() < 0.1 # 10%的几率加traj
        has_text_lst = (audio_length == 0) # 没 audio 的加 text
        has_text_lst[has_text_lst == False] = random() < 0.1 # audio 10% 的几率加 text

        text_feat, text_mask = self.text_encoder(text_data)

        traj_feat = self.traj_encoder(traj_data.permute(0, 2, 1)).permute(0, 2, 1)
        traj_mask = length_to_mask(motion_length // self.motion_downsample_ratio)
        traj_mask[:] = False if not has_traj else True

        text_mask = text_mask * has_text_lst[:, None]

        if not torch.all(audio_length == 0):
            audio_feat = self.audio_encoder(audio_data.permute(0, 2, 1)).permute(0, 2, 1)
            audio_mask = length_to_mask(audio_length // self.audio_downsample_ratio)
        else:
            # audio下采样8倍
            audio_data = torch.zeros((B, 8, 512)).to(motion)
            audio_feat = self.audio_encoder(audio_data.permute(0, 2, 1)).permute(0, 2, 1).mean() * 0.0
            audio_mask = torch.zeros(B, 0).to(text_feat.device)

            text_feat += audio_feat * 0.0
            traj_feat += audio_feat * 0.0
            audio_feat = torch.zeros(B, 0, self.n_embd).to(text_feat.device)

        if mode == 'loss':
            tokens, label_tokens, non_pad_mask = self.prepare_train_input(motion, motion_length)
            loss = self.forward_loss(
                tokens, label_tokens, text_feat, text_mask, audio_feat, 
                audio_mask, traj_feat, traj_mask, non_pad_mask, motion=motion, 
                mean=kwargs['mean'], std=kwargs['std']
            )
            return loss
        else:
            # 只测text
            text_feat, text_mask = self.text_encoder(text_data)
            B = text_feat.shape[0]
            audio_feat = torch.zeros(B, 0, self.n_embd).to(device)
            audio_mask = torch.zeros(B, 0).to(device)
            traj_feat = torch.zeros(B, 0, self.n_embd).to(device)
            traj_mask = torch.zeros(B, 0).to(device)

            non_pad_mask = self.prepare_test_input(motion_length)
            return self.forward_predict(text_feat, text_mask, audio_feat, audio_mask, traj_feat, traj_mask, non_pad_mask, **kwargs)
    
    def forward(
            self, 
            motion: torch.Tensor, 
            motion_length: torch.Tensor,

            text_data: List[str]=None,
            audio_data: torch.Tensor=None,
            traj_data: torch.Tensor=None,
            audio_length: torch.Tensor=None,
            
            mode = 'loss', 
            **kwargs
        ): 
        '''
        motion: tensor, B, T, C
        motion_length: tensor, B
        text_data: list[str], len B
        has_xxx: list[bool], len B
        music_data: tensor, B, T, C
        '''
        if self.stage == 1:
            return self.forward_stage_1(motion, motion_length, text_data, mode)
        elif self.stage == 2:
            return self.forward_stage_2(motion, motion_length, audio_data, traj_data, audio_length, mode, **kwargs)
        elif self.stage == 3:
            return self.forward_stage_3(motion, motion_length, text_data, audio_data, traj_data, audio_length, mode, **kwargs)
        elif self.stage == 4: # for evaluation
            B = motion.shape[0]
            device = motion.device
            text_feat = torch.zeros(B, 0, self.n_embd).to(device)
            audio_feat = torch.zeros(B, 0, self.n_embd).to(device)
            traj_feat = torch.zeros(B, 0, self.n_embd).to(device)
            text_mask = torch.zeros(B, 0).to(device)
            audio_mask = torch.zeros(B, 0).to(device)
            traj_mask = torch.zeros(B, 0).to(device)

            if 'text' in self.cond_types:
                text_feat, text_mask = self.text_encoder(text_data)
                dtype = text_feat.dtype
            if 'audio' in self.cond_types:
                audio_feat = self.audio_encoder(audio_data.permute(0, 2, 1)).permute(0, 2, 1)
                audio_mask = length_to_mask(audio_length // self.audio_downsample_ratio)
                dtype = audio_feat.dtype
            if 'traj' in self.cond_types:
                traj_feat = self.traj_encoder(traj_data.permute(0, 2, 1)).permute(0, 2, 1)
                traj_mask = length_to_mask(motion_length // self.motion_downsample_ratio)
                dtype = traj_feat.dtype

            text_feat, text_mask = text_feat.to(dtype), text_mask.to(device)
            audio_feat, audio_mask = audio_feat.to(dtype), audio_mask.to(dtype)
            traj_feat, traj_mask = traj_feat.to(dtype), traj_mask.to(dtype)

            non_pad_mask = self.prepare_test_input(motion_length)
            return self.forward_predict(text_feat, text_mask, audio_feat, audio_mask, traj_feat, traj_mask, non_pad_mask, **kwargs)
        else:
            raise NotImplementedError
    
    def prepare_train_input(self, motion, motion_length):
        if self.vqvae.training:
            self.vqvae.eval()

        B = motion.shape[0]
        device = motion.device
        m_token_lens = motion_length // self.motion_downsample_ratio
        max_token_len = m_token_lens.max()
        ids = self.vqvae.encode(motion, motion_length) # B, T, n_quant
        # True, True, ..., False
        non_pad_mask = torch.arange(max_token_len, device=device).expand(B, max_token_len) < m_token_lens.unsqueeze(1)

        all_x_ids = []
        all_labels = []
        mask = None
        for i in range(ids.shape[-1]): # n_quant
            m_tokens_i = ids[..., i] # B, T

            # 1. prepare label token
            labels = torch.where(non_pad_mask, m_tokens_i, self.ignore_index)

            # 2. prepare input token
            m_tokens_i = torch.where(non_pad_mask, m_tokens_i, self.pad_idx) # [m1, m2, m3, ..., pad1, pad2, ...]
            
            # 2.1 prepare mask
            if mask is None:
                rand_time = uniform((B,), device=device)
                rand_mask_probs = self.noise_schedule(rand_time)
                num_token_masked = (max_token_len * rand_mask_probs).round().clamp(min=1)
                batch_randperm = torch.rand((B, max_token_len), device=device).argsort(dim=-1)
                # Positions to be masked are all true
                mask = batch_randperm < num_token_masked.unsqueeze(-1)
                # Positions to be MASKED must also be NON-PADDED
                mask &= non_pad_mask

            # 2.2 Further Apply Bert Masking Scheme
            # Step 1: 10% replace with an incorrect token
            x_ids = m_tokens_i.clone()
            mask_rid = get_mask_subset_prob(mask, 0.1)
            rand_id = torch.randint_like(x_ids, high=self.vocab_size - 2)
            x_ids = torch.where(mask_rid, rand_id, x_ids)
            # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
            mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
            x_ids = torch.where(mask_mid, self.mask_idx, x_ids)
            all_x_ids.append(x_ids)
            all_labels.append(labels)

        return all_x_ids, all_labels, non_pad_mask

    def forward_loss(self, 
                     tokens, label_tokens, 
                     text_feat, text_mask, 
                     audio_feat, audio_mask, 
                     traj_feat, traj_mask, 
                     non_pad_mask, mean=None, std=None, motion=None):
        all_logits = self.transformer(tokens, non_pad_mask, text_feat, text_mask, audio_feat, audio_mask, traj_feat, traj_mask)

        ce_loss = torch.tensor(0.).to(text_feat)
        acc_dict = dict()

        for i, (logits, labels) in enumerate(zip(all_logits, label_tokens)):
            logits = logits.flatten(0, 1)
            labels = labels.flatten(0, 1)
            ce_loss = ce_loss + F.cross_entropy(logits, labels.to(torch.int64), ignore_index=self.ignore_index)

            # calculate acc
            probs = torch.softmax(logits, dim=-1)
            cls_pred_index = torch.argmax(probs, dim=-1)
            token_mask = (labels != self.ignore_index).int()
            right_num = (cls_pred_index == labels).int() * token_mask
            acc_dict[f'ACC_{i}'] = (right_num.sum() / token_mask.sum()).float().detach()

        # add traj loss
        traj_loss = torch.tensor(0.0)
        if traj_mask.sum() > 0:
            all_pred_motion_tokens = []
            for i, (logits, labels) in enumerate(zip(all_logits, label_tokens)):
                probs_gumbel_softmax = F.gumbel_softmax(logits, tau=1, eps=1e-10, hard=True, dim = -1)
                pred_motion_tokens = torch.topk(probs_gumbel_softmax, k=1, dim=-1)[1].reshape(labels.shape[0], -1) # B, T
                all_pred_motion_tokens.append(pred_motion_tokens)
            all_pred_motion_tokens = torch.stack(all_pred_motion_tokens).permute(1, 2, 0).to(logits.dtype) # nq, B, T

            pred_motion = self.vqvae.decode(all_pred_motion_tokens, motion_length=non_pad_mask.sum(1))

            pred_traj = recover_from_local_position_fast(pred_motion.float() * std + mean, njoint=22)[:, :, 0].to(logits.dtype)
            gt_traj = recover_from_local_position_fast(motion.float() * std + mean, njoint=22)[:, :, 0].to(logits.dtype)

            arange_t = torch.arange(motion.shape[1])[None, :].to(motion.device)  # (1, T)
            mask = arange_t < (non_pad_mask.sum(1)[:, None] * self.motion_downsample_ratio)       # (B, T)
            loss_raw = F.l1_loss(pred_traj, gt_traj, reduction='none') # (B, T, C)
            traj_loss = (loss_raw * mask[..., None]).sum() / (mask.sum() * loss_raw.shape[-1])
        return dict(
            ce_loss=ce_loss / self.n_quant, 
            traj_loss=traj_loss, 
            **acc_dict
        )
    
    @torch.no_grad()
    def forward_predict(self, text_feat, text_mask, audio_feat, audio_mask, traj_feat, traj_mask, non_pad_mask, return_ids=False, **kwargs):
        device = non_pad_mask.device
        # tokens: all pad mask seqence
        ids = torch.where(non_pad_mask, self.mask_idx, self.pad_idx)[None].repeat(self.n_quant, 1, 1)
        scores = torch.where(non_pad_mask, 0., 1e5)[None].repeat(self.n_quant, 1, 1)
        m_lens = non_pad_mask.sum(-1)

        for timestep in torch.linspace(0, 1, self.inference_timesteps, device=device):
            rand_mask_prob = self.noise_schedule(timestep)
            # Maskout, and cope with variable length
            num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)  # (b, )
            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(dim=-1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=-1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1).unsqueeze(0))
            ids = torch.where(is_mask, self.mask_idx, ids)
            # all_logits = self.transformer(ids, non_pad_mask, text_feat, text_mask)
            all_logits = self.transformer(ids, non_pad_mask, text_feat, text_mask, audio_feat, audio_mask, traj_feat, traj_mask)
            
            for i, logits in enumerate(all_logits):
                filtered_logits = top_k(logits, self.topk_filter_thres, dim=-1)
                probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)

                ids[i] = torch.where(is_mask[i], pred_ids, ids[i])

                # Updating scores
                probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
                scores[i] = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1)).squeeze(-1)  # (b, seqlen)

                # We do not want to re-mask the previously kept tokens, or pad tokens
                scores[i] = scores[i].masked_fill(~is_mask[i], 1e5)

        ids = torch.where(non_pad_mask[None].repeat(self.n_quant, 1, 1), ids, -1)
        pred_motions = []
        pred_motion_lengths = []
        
        pred_motions = self.vqvae.decode(ids.permute(1, 2, 0), motion_length=non_pad_mask.sum(1))

        if return_ids:
            return pred_motions, pred_motion_lengths, ids
        else:
            return pred_motions

    def prepare_test_input(self, motion_length):
        B = motion_length.shape[0]
        device = motion_length.device
        m_lens = motion_length // self.motion_downsample_ratio
        max_len = motion_length.max() // self.motion_downsample_ratio
        non_pad_mask = torch.arange(max_len, device=device).expand(B, max_len) < m_lens.unsqueeze(1)
        return non_pad_mask

@MODELS.register_module()
class AnyMo_audio(BaseModel):
    def __init__(
            self, 
            vqvae_cfg: Dict,
            transformer_cfg: Dict,

            max_motion_len: int,
            motion_downsample_ratio: int, 
            audio_downsample_ratio: int, 

            text_encoder_cfg: Dict=None,
            traj_encoder_cfg: Dict=None, 
            audio_encoder_cfg: Dict=None,
            
            inference_timesteps=6, 
            cond_types = ['audio'], 
            **kwargs
    ):
        super().__init__(**kwargs)

        self.transformer = MODELS.build(transformer_cfg)
        self.text_encoder = MODELS.build(text_encoder_cfg)
        self.traj_encoder = MODELS.build(traj_encoder_cfg)
        self.audio_encoder = MODELS.build(audio_encoder_cfg)

        # build vqvae
        self.vqvae = MODELS.build(vqvae_cfg)
        self.vqvae.init_weights()
        for param in self.vqvae.parameters():
            param.requires_grad = False
        self.vqvae.eval()

        self.vocab_size = transformer_cfg.vocab_size # type: ignore
        self.n_embd = transformer_cfg.n_embd
        self.mask_idx = self.vocab_size - 1
        self.pad_idx = self.vocab_size - 2
        self.ignore_index = -100

        self.max_motion_len = max_motion_len
        self.motion_downsample_ratio = motion_downsample_ratio
        self.audio_downsample_ratio = audio_downsample_ratio
        self.topk_filter_thres = 0.9
        self.noise_schedule = cosine_schedule
        self.n_quant = len(self.vqvae.quantizer.quantizers)
        self.inference_timesteps = inference_timesteps
        self.cond_types = cond_types
        for cond_type in cond_types:
            assert cond_type in ['text', 'audio', 'traj']

    def forward_train(self, motion, motion_length, text_data, audio_data, traj_data, audio_length, **kwargs):
        audio_feat = self.audio_encoder(audio_data.permute(0, 2, 1)).permute(0, 2, 1)
        audio_mask = length_to_mask(audio_length // self.audio_downsample_ratio)

        B = motion.shape[0]
        device = motion.device
        text_feat = torch.zeros(B, 0, self.n_embd).to(device).to(audio_feat.dtype)
        traj_feat = torch.zeros(B, 0, self.n_embd).to(device).to(audio_feat.dtype)
        text_mask = torch.zeros(B, 0).to(device).to(audio_feat.dtype)
        traj_mask = torch.zeros(B, 0).to(device).to(audio_feat.dtype)

        tokens, label_tokens, non_pad_mask = self.prepare_train_input(motion, motion_length)
        loss =  self.forward_loss(
            tokens, label_tokens, text_feat, text_mask, audio_feat, 
            audio_mask, traj_feat, traj_mask, non_pad_mask, motion=motion, mean=kwargs['mean'], 
            std=kwargs['std']
        )
        return loss
    
    def forward(
            self, 
            motion: torch.Tensor, 
            motion_length: torch.Tensor,

            text_data: List[str]=None,
            audio_data: torch.Tensor=None,
            traj_data: torch.Tensor=None,
            audio_length: torch.Tensor=None,
            
            mode = 'loss', 
            **kwargs
        ): 
        '''
        motion: tensor, B, T, C
        motion_length: tensor, B
        text_data: list[str], len B
        has_xxx: list[bool], len B
        music_data: tensor, B, T, C
        '''
        if mode == 'loss':
            return self.forward_train(motion, motion_length, text_data, audio_data, traj_data, audio_length, **kwargs)
        else: # for evaluation
            B = motion.shape[0]
            device = motion.device
            text_feat = torch.zeros(B, 0, self.n_embd).to(device)
            audio_feat = torch.zeros(B, 0, self.n_embd).to(device)
            traj_feat = torch.zeros(B, 0, self.n_embd).to(device)
            text_mask = torch.zeros(B, 0).to(device)
            audio_mask = torch.zeros(B, 0).to(device)
            traj_mask = torch.zeros(B, 0).to(device)
            if 'text' in self.cond_types:
                text_feat, text_mask = self.text_encoder(text_data)
                dtype = text_feat.dtype
            if 'audio' in self.cond_types:
                audio_feat = self.audio_encoder(audio_data.permute(0, 2, 1)).permute(0, 2, 1)
                audio_mask = length_to_mask(audio_length // self.audio_downsample_ratio)
                dtype = audio_feat.dtype
            if 'traj' in self.cond_types:
                traj_feat = self.traj_encoder(traj_data.permute(0, 2, 1)).permute(0, 2, 1)
                traj_mask = length_to_mask(motion_length // self.motion_downsample_ratio)
                dtype = traj_feat.dtype

            text_feat, text_mask = text_feat.to(dtype), text_mask.to(device)
            audio_feat, audio_mask = audio_feat.to(dtype), audio_mask.to(dtype)
            traj_feat, traj_mask = traj_feat.to(dtype), traj_mask.to(dtype)

            non_pad_mask = self.prepare_test_input(motion_length)
            return self.forward_predict(text_feat, text_mask, audio_feat, audio_mask, traj_feat, traj_mask, non_pad_mask, **kwargs)
    
    def prepare_train_input(self, motion, motion_length):
        if self.vqvae.training:
            self.vqvae.eval()

        B = motion.shape[0]
        device = motion.device
        m_token_lens = motion_length // self.motion_downsample_ratio
        max_token_len = m_token_lens.max()
        ids = self.vqvae.encode(motion, motion_length) # B, T, n_quant
        # True, True, ..., False
        non_pad_mask = torch.arange(max_token_len, device=device).expand(B, max_token_len) < m_token_lens.unsqueeze(1)

        all_x_ids = []
        all_labels = []
        mask = None
        for i in range(ids.shape[-1]): # n_quant
            m_tokens_i = ids[..., i] # B, T

            # 1. prepare label token
            labels = torch.where(non_pad_mask, m_tokens_i, self.ignore_index)

            # 2. prepare input token
            m_tokens_i = torch.where(non_pad_mask, m_tokens_i, self.pad_idx) # [m1, m2, m3, ..., pad1, pad2, ...]
            
            # 2.1 prepare mask
            if mask is None:
                rand_time = uniform((B,), device=device)
                rand_mask_probs = self.noise_schedule(rand_time)
                num_token_masked = (max_token_len * rand_mask_probs).round().clamp(min=1)
                batch_randperm = torch.rand((B, max_token_len), device=device).argsort(dim=-1)
                # Positions to be masked are all true
                mask = batch_randperm < num_token_masked.unsqueeze(-1)
                # Positions to be MASKED must also be NON-PADDED
                mask &= non_pad_mask

            # 2.2 Further Apply Bert Masking Scheme
            # Step 1: 10% replace with an incorrect token
            x_ids = m_tokens_i.clone()
            mask_rid = get_mask_subset_prob(mask, 0.1)
            rand_id = torch.randint_like(x_ids, high=self.vocab_size - 2)
            x_ids = torch.where(mask_rid, rand_id, x_ids)
            # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
            mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
            x_ids = torch.where(mask_mid, self.mask_idx, x_ids)
            all_x_ids.append(x_ids)
            all_labels.append(labels)

        return all_x_ids, all_labels, non_pad_mask

    def forward_loss(self, 
                     tokens, label_tokens, 
                     text_feat, text_mask, 
                     audio_feat, audio_mask, 
                     traj_feat, traj_mask, 
                     non_pad_mask, mean=None, std=None, motion=None):
        all_logits = self.transformer(tokens, non_pad_mask, text_feat, text_mask, audio_feat, audio_mask, traj_feat, traj_mask)

        ce_loss = torch.tensor(0.).to(text_feat)
        acc_dict = dict()

        for i, (logits, labels) in enumerate(zip(all_logits, label_tokens)):
            logits = logits.flatten(0, 1)
            labels = labels.flatten(0, 1)
            ce_loss = ce_loss + F.cross_entropy(logits, labels.to(torch.int64), ignore_index=self.ignore_index)

            # calculate acc
            probs = torch.softmax(logits, dim=-1)
            cls_pred_index = torch.argmax(probs, dim=-1)
            token_mask = (labels != self.ignore_index).int()
            right_num = (cls_pred_index == labels).int() * token_mask
            acc_dict[f'ACC_{i}'] = (right_num.sum() / token_mask.sum()).float().detach()

        # add traj loss
        traj_loss = torch.tensor(0.0)
        if traj_mask.sum() > 0:
            all_pred_motion_tokens = []
            for i, (logits, labels) in enumerate(zip(all_logits, label_tokens)):
                probs_gumbel_softmax = F.gumbel_softmax(logits, tau=1, eps=1e-10, hard=True, dim = -1)
                pred_motion_tokens = torch.topk(probs_gumbel_softmax, k=1, dim=-1)[1].reshape(labels.shape[0], -1) # B, T
                all_pred_motion_tokens.append(pred_motion_tokens)
            all_pred_motion_tokens = torch.stack(all_pred_motion_tokens) # nq, B, T
            pred_motion = self.vqvae.decode(all_pred_motion_tokens.permute(1, 2, 0), motion_length=non_pad_mask.sum(1))

            pred_traj = recover_from_local_position_fast(pred_motion * std + mean, njoint=22)[:, :, 0]
            gt_traj = recover_from_local_position_fast(motion * std + mean, njoint=22)[:, :, 0]

            arange_t = torch.arange(motion.shape[1])[None, :].to(motion.device)  # (1, T)
            mask = arange_t < (non_pad_mask.sum(1)[:, None] * self.motion_downsample_ratio)       # (B, T)
            loss_raw = F.l1_loss(pred_traj, gt_traj, reduction='none') # (B, T, C)
            traj_loss = (loss_raw * mask[..., None]).sum() / (mask.sum() * loss_raw.shape[-1])

        return dict(
            ce_loss=ce_loss / self.n_quant, 
            traj_loss=traj_loss, 
            **acc_dict
        )
    
    @torch.no_grad()
    def forward_predict(self, text_feat, text_mask, audio_feat, audio_mask, traj_feat, traj_mask, non_pad_mask, return_ids=False, **kwargs):
        device = non_pad_mask.device
        # tokens: all pad mask seqence
        ids = torch.where(non_pad_mask, self.mask_idx, self.pad_idx)[None].repeat(self.n_quant, 1, 1)
        scores = torch.where(non_pad_mask, 0., 1e5)[None].repeat(self.n_quant, 1, 1)
        m_lens = non_pad_mask.sum(-1)

        for timestep in torch.linspace(0, 1, self.inference_timesteps, device=device):
            rand_mask_prob = self.noise_schedule(timestep)
            # Maskout, and cope with variable length
            num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)  # (b, )
            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(dim=-1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=-1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1).unsqueeze(0))
            ids = torch.where(is_mask, self.mask_idx, ids)
            # all_logits = self.transformer(ids, non_pad_mask, text_feat, text_mask)
            all_logits = self.transformer(ids, non_pad_mask, text_feat, text_mask, audio_feat, audio_mask, traj_feat, traj_mask)
            
            for i, logits in enumerate(all_logits):
                filtered_logits = top_k(logits, self.topk_filter_thres, dim=-1)
                probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)

                ids[i] = torch.where(is_mask[i], pred_ids, ids[i])

                # Updating scores
                probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
                scores[i] = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1)).squeeze(-1)  # (b, seqlen)

                # We do not want to re-mask the previously kept tokens, or pad tokens
                scores[i] = scores[i].masked_fill(~is_mask[i], 1e5)

        ids = torch.where(non_pad_mask[None].repeat(self.n_quant, 1, 1), ids, -1)
        pred_motions = []
        pred_motion_lengths = []
        
        pred_motions = self.vqvae.decode(ids.permute(1, 2, 0), motion_length=non_pad_mask.sum(1))

        if return_ids:
            return pred_motions, pred_motion_lengths, ids
        else:
            return pred_motions

    def prepare_test_input(self, motion_length):
        B = motion_length.shape[0]
        device = motion_length.device
        m_lens = motion_length // self.motion_downsample_ratio
        max_len = motion_length.max() // self.motion_downsample_ratio
        non_pad_mask = torch.arange(max_len, device=device).expand(B, max_len) < m_lens.unsqueeze(1)
        return non_pad_mask
