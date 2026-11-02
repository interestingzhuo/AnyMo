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
from src.utils.audio_feat.kinetic import extract_kinetic_features
from src.utils.audio_feat.manual_new import extract_manual_features
from ..utils.motion import recover_from_local_rotation

def vis_motion(motion, mean, std):
    from .recons_metric import recover_from_local_position
    from ..utils.visualize import vis_3d_motion
    motion = (motion * std + mean).numpy()
    positions = recover_from_local_position(motion, 22) # 
    vis_3d_motion(positions[None], None, [f'video.mp4'], fps=30)

@METRICS.register_module()
class Music2DanceMetric(BaseMetric):
    def __init__(self, gt_feature_path="", **kwargs):
        super().__init__(**kwargs)
        self.gt_feature = joblib.load(gt_feature_path)

    @torch.no_grad()
    def process(self, data_batch, data_samples):
        pred_motions = data_samples
        
        pred_motions = pad_sequence(pred_motions, batch_first=True)
        gt_motions = pad_sequence(data_batch['motion'], batch_first=True).cuda()
        motion_lengths = data_batch['motion_length']
        mean, std = data_batch['mean'].cuda(), data_batch['std'].cuda()
        captions = data_batch['text_data']

        # ======================================================================
        # idx = 1; vis_motion(data_batch['motion'][idx][:motion_lengths[idx]], mean.cpu(), std.cpu())
        # idx = -1; vis_motion(pred_motions[idx][:motion_lengths[idx]].cpu(), mean.cpu(), std.cpu())
        # idx = -6;pred_data = recover_from_local_rotation((pred_motions * std + mean).cpu().numpy()[idx][:motion_lengths[idx]], njoint=22); np.savez('tmp.npz', **pred_data)
        # ======================================================================

        pred_motions = pred_motions * std + mean
        gt_motions = gt_motions * std + mean

        pred_positions = recover_from_local_position_fast(pred_motions, 22) # B, n, 22 ,3
        pred_positions = pred_positions - pred_positions[:, :1, :1]

        gt_positions = recover_from_local_position_fast(gt_motions, 22)
        gt_positions = gt_positions - gt_positions[:, :1, :1]

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
                continue

            try:
                gt_beat_align_score = BA(music_beat_axis, gt_joint_beat)
            except:
                continue
            
            # pred joint feat
            pred_kinetic_features = extract_kinetic_features(pred_positions[i, :m_len].cpu().numpy())
            pred_manual_features_new = extract_manual_features(pred_positions[i, :m_len].cpu().numpy())
            # gt_manual_features_new = extract_manual_features(gt_positions[i, :m_len].cpu().numpy())
            # gt_kinetic_features = extract_kinetic_features(gt_positions[i, :m_len].cpu().numpy())
        
            self.results.append({
                'pred_beat_align_score': pred_beat_align_score,
                'gt_beat_align_score': gt_beat_align_score,
                'pred_kinetic_features': pred_kinetic_features, 
                # 'gt_kinetic_features': gt_kinetic_features, 
                'pred_manual_features_new': pred_manual_features_new, 
                # 'gt_manual_features_new': gt_manual_features_new, 
            })

    def compute_metrics(self, results):
        pred_beat_align_score = np.array([item['pred_beat_align_score'] for item in results]).mean()
        gt_beat_align_score = np.array([item['gt_beat_align_score'] for item in results]).mean()
        pred_features_k = np.stack([item['pred_kinetic_features'] for item in results]) # Nx72 p40
        pred_features_m = np.stack([item['pred_manual_features_new'] for item in results]) # Nx32
        gt_freatures_k = np.stack(self.gt_feature['all_gt_kinetic_features'])
        # gt_freatures_k = np.stack([item['gt_kinetic_features'] for item in results]) # Nx72 p40
        gt_freatures_m = np.stack(self.gt_feature['all_gt_manual_features_new']) 
        # gt_freatures_m = np.stack([item['gt_manual_features_new'] for item in results]) # Nx32

        gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
        gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m) 
        fid_k = calc_fid(pred_features_k, gt_freatures_k)
        fid_m = calc_fid(pred_features_m, gt_freatures_m)
        # div_k_gt = calculate_avg_distance(gt_freatures_k)
        # div_m_gt = calculate_avg_distance(gt_freatures_m)
        div_k = calculate_avg_distance(pred_features_k)
        div_m = calculate_avg_distance(pred_features_m)
        return dict(
            pred_beat_align_score=pred_beat_align_score, 
            gt_beat_align_score=gt_beat_align_score, 
            fid_k=fid_k, 
            fid_m=fid_m, 
            div_k=div_k, 
            div_m=div_m, 
            # div_k_gt=div_k_gt,
            # div_m_gt=div_m_gt, 
        )

def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))


def calc_joint_beat(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 22, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

def calc_fid(kps_gen, kps_gt):
    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)
    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)
    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # 处理虚数
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist