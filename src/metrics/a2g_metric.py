import joblib
import torch
import numpy as np
from scipy import linalg
from torch.nn.utils.rnn import pad_sequence
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from .recons_metric import recover_from_local_position
from src.utils.motion import recover_from_local_position_fast
from mmengine import Config
from mmengine.registry import MODELS
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
from .m2d_metric import calc_joint_beat, BA
from .t2m_metric import calculate_activation_statistics, calculate_diversity, calculate_frechet_distance
from ..utils.motion import recover_from_local_rotation

def vis_motion(motion, mean, std):
    from .recons_metric import recover_from_local_position
    from ..utils.visualize import vis_3d_motion
    motion = (motion * std + mean).numpy()
    positions = recover_from_local_position(motion, 22) # 
    vis_3d_motion(positions[None], None, [f'video.mp4'], fps=30)

@METRICS.register_module()
class Audio2MotionMetric(BaseMetric):
    def __init__(self, tmr_cfg_path='', tmr_ckpt_path='', **kwargs):
        super().__init__(**kwargs)
        cfg = Config.fromfile(tmr_cfg_path)
        self.tmr_model = MODELS.build(cfg.model)
        ckpt = torch.load(tmr_ckpt_path, weights_only=False, map_location='cpu')
        self.tmr_model.load_state_dict(ckpt['state_dict'])
        self.tmr_model.cuda().eval()

    @torch.no_grad()
    def process(self, data_batch, data_samples):
        pred_motions = data_samples
        pred_motions = pad_sequence(pred_motions, batch_first=True)
        gt_motions = pad_sequence(data_batch['motion'], batch_first=True).cuda()
        motion_lengths = data_batch['motion_length']
        mean, std = data_batch['mean'].cuda(), data_batch['std'].cuda()
        captions = data_batch['text_data']

        pad_m = torch.zeros((pred_motions.shape[0], 300 - motion_lengths.max(), 272)).to(pred_motions)
        pred_motions = torch.cat([pred_motions, pad_m], dim=1)
        gt_motions = torch.cat([gt_motions, pad_m], dim=1)
        gt_m_latents = self.tmr_model.motionencoder(gt_motions.cuda(), motion_lengths.cuda()).loc
        m_latents = self.tmr_model.motionencoder(pred_motions.cuda(), motion_lengths.cuda()).loc

        pred_positions = recover_from_local_position_fast(pred_motions * std + mean, 22) # B, n, 22 ,3
        pred_positions = pred_positions - pred_positions[:, :1, :1]

        gt_positions = recover_from_local_position_fast(gt_motions * std + mean, 22)
        gt_positions = gt_positions - gt_positions[:, :1, :1]

        bas = []
        gt_bas = []
        flag = []
        for i in range(gt_positions.shape[0]):
            # music beat
            m_len = motion_lengths[i]
            music_beat = data_batch['audio_beat'][i].to(torch.bool).numpy()[:m_len]
            music_beat_axis = np.arange(len(music_beat))[music_beat]

            # BAS
            pred_joint_beat = calc_joint_beat(pred_positions[i, :m_len].cpu())
            gt_joint_beat = calc_joint_beat(gt_positions[i, :m_len].cpu())
            try:
                pred_beat_align_score = BA(music_beat_axis, pred_joint_beat)
            except:
                # print(pred_joint_beat)
                pred_beat_align_score = 0.0

            try:
                gt_beat_align_score = BA(music_beat_axis, gt_joint_beat)
            except:
                # print(gt_joint_beat)
                gt_beat_align_score = 0.0

            bas.append(pred_beat_align_score)
            gt_bas.append(gt_beat_align_score)

        self.results.append({
            'bas': bas, 
            'gt_bas': gt_bas, 
            'pred_motion_embd': m_latents, 
            'gt_motion_embd': gt_m_latents, 
            'samplt_cnt': len(captions)
        })

    def compute_metrics(self, results):
        count = sum(item['samplt_cnt'] for item in results)
        pred_motion_embd = np.concatenate([item['pred_motion_embd'].numpy() for item in results], axis=0)
        gt_motion_embd = np.concatenate([item['gt_motion_embd'].numpy() for item in results], axis=0)
        pred_bas = np.array([bas for item in results for bas in item['bas']])
        gt_bas = np.array([bas for item in results for bas in item['gt_bas']])

        idx = np.logical_and(pred_bas != 0, gt_bas != 0)
        pred_bas = pred_bas[idx].mean()
        gt_bas = gt_bas[idx].mean()
        pred_motion_embd = pred_motion_embd[idx]
        gt_motion_embd = gt_motion_embd[idx]

        gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embd)
        mu, cov = calculate_activation_statistics(pred_motion_embd)

        diversity_real = calculate_diversity(gt_motion_embd, 300 if count > 300 else 100)
        diversity = calculate_diversity(pred_motion_embd, 300 if count > 300 else 100)

        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        
        return dict(
            fid=fid,
            div_real=diversity_real, 
            div=diversity, 
            gt_bas=gt_bas, 
            bas=pred_bas, 
        )
