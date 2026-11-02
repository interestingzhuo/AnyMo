import torch
import numpy as np
from scipy import linalg
from torch.nn.utils.rnn import pad_sequence
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from .recons_metric import recover_from_local_position
from ..utils.motion import recover_from_local_rotation
from mmengine import Config
from mmengine.registry import MODELS

def vis_motion(motion, mean, std):
    from .recons_metric import recover_from_local_position
    from ..utils.visualize import vis_3d_motion
    motion = (motion * std + mean).numpy()
    positions = recover_from_local_position(motion, 22) # 
    vis_3d_motion(positions[None], None, [f'video.mp4'], fps=30)

def euclidean_distance_matrix(matrix1, matrix2):
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    
    d3 = np.sum(np.square(matrix2), axis=1)     
    dists = np.sqrt(d1 + d2 + d3)
    return dists

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1) 
    mu2 = np.atleast_1d(mu2) 

    sigma1 = np.atleast_2d(sigma1) 
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False) 
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)  
    return top_k_mat

def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace() 
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)  
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1) 
    return dist.mean()  

def compute_kps_error(cur_motion, gt_skel_motions, original_mask):
    '''
    cur_motion [bs, 22, 3, seq_len]
    gt_skel_motions [bs, 22, 3, seq_len]
    mask [bs, 22, 3, seq_len]
    '''
    mask = original_mask.permute(0, 3, 1, 2).bool().float().sum(dim=-1).clamp(0,1)  # [bs, 22, 3, seq_len] -> [bs, seq_len, 22, 3] -> [bs, seq_len, 22]
    joint_error = (cur_motion.permute(0, 3, 1, 2) - gt_skel_motions.permute(0, 3, 1, 2)) 
    assert joint_error.shape[-1] == 3
    dist_err = torch.linalg.norm(joint_error, dim=-1) * mask.float() # [bs, seq_len, 22]  joint error norm in xyz dim
    mask = mask.sum(dim=-1) #[bs, seq_len]
    dist_err = dist_err.sum(dim=-1) # [bs, seq_len]
    dist_err = torch.where(mask > 0, dist_err / mask, torch.tensor(0.0, dtype=torch.float32, device=dist_err.device)) # [bs, seq_len]   joint error average on 22 joints
    mask = mask.clamp(0,1)  # [bs, seq_len]
    return dist_err.detach().cpu().numpy(), mask.detach().cpu().numpy()

def calculate_trajectory_error(dist_error, mask, strict=True):
    ''' dist_error and mask shape [bs, seq_len]: error for each kps in metre
      Two threshold: 20 cm and 50 cm.
    If mean error in sequence is more then the threshold, fails
    return: traj_fail(0.2), traj_fail(0.5), all_kps_fail(0.2), all_kps_fail(0.5), all_mean_err.
        Every metrics are already averaged.
    '''
    assert dist_error.shape == mask.shape
    bs, seqlen = dist_error.shape
    assert np.all(np.logical_or(mask == 0, mask == 1)), "mask array contains values other than 0 or 1"
    mask_sum = mask.sum(-1)  # [bs]
    mean_err_traj = np.where(mask_sum > 1e-5, (dist_error * mask.astype(float)).sum(-1)/ mask_sum, 0.0)

    if strict:
        # Traj fails if any of the key frame fails
        traj_fail_02 = 1.0 - (dist_error * mask.astype(float) <= 0.2).all(1)
        traj_fail_05 = 1.0 - (dist_error * mask.astype(float) <= 0.5).all(1)
    else:
        # Traj fails if the mean error of all keyframes more than the threshold
        traj_fail_02 = (mean_err_traj > 0.2).astype(float)
        traj_fail_05 = (mean_err_traj > 0.5).astype(float)
    all_fail_02 = np.where(mask_sum > 1e-5, (dist_error * mask.astype(float) > 0.2).sum(-1) / mask_sum, 0.0)
    all_fail_05 = np.where(mask_sum > 1e-5, (dist_error * mask.astype(float) > 0.5).sum(-1) / mask_sum, 0.0)
    result = np.concatenate([traj_fail_02[..., None], traj_fail_05[..., None], all_fail_02[..., None], all_fail_05[..., None], mean_err_traj[..., None]], axis=1)
    
    assert np.all(~np.isnan(result)), "result array contains NaN values"
    assert result.shape == (bs, 5)
    return result


@METRICS.register_module()
class Text2MotionMetric(BaseMetric):
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

        # ===================================
        # idx = 11; vis_motion(data_batch['motion'][idx][:motion_lengths[idx]], mean.cpu(), std.cpu())
        # idx = 12; vis_motion(pred_motions[idx][:motion_lengths[idx]].cpu(), mean.cpu(), std.cpu())
        # ===================================
        
        pad_m = torch.zeros((pred_motions.shape[0], 300 - motion_lengths.max(), 272)).to(pred_motions)
        pred_motions = torch.cat([pred_motions, pad_m], dim=1)
        gt_motions = torch.cat([gt_motions, pad_m], dim=1)

        with torch.no_grad():
            gt_m_latents = self.tmr_model.motionencoder(gt_motions.cuda(), motion_lengths.cuda()).loc
            m_latents = self.tmr_model.motionencoder(pred_motions.cuda(), motion_lengths.cuda()).loc
            t_latents = self.tmr_model.textencoder(captions).loc

        # _, gt_m_latents, _ = self.tmr_model(gt_motions.cuda(), motion_lengths.cuda(), captions, mode='predict')
        # t_latents, m_latents, sent_emb = self.tmr_model(pred_motions.cuda(), motion_lengths.cuda(), captions, mode='predict')

        R_precision_real, matching_score_real = calculate_R_precision(t_latents.cpu().numpy(), gt_m_latents.cpu().numpy(), top_k=3, sum_all=True)
        R_precision, matching_score_pred = calculate_R_precision(t_latents.cpu().numpy(), m_latents.cpu().numpy(), top_k=3, sum_all=True)

        self.results.append({
            'R_precision_real': R_precision_real,
            'matching_score_real': matching_score_real, 
            'R_precision': R_precision, 
            'matching_score_pred': matching_score_pred, 
            'pred_motion_embd': m_latents, 
            'gt_motion_embd': gt_m_latents, 
            'samplt_cnt': len(captions)
        })

    def compute_metrics(self, results):
        count = sum(item['samplt_cnt'] for item in results)
        R_precision_real = np.array([item['R_precision_real'] for item in results]).sum(axis=0) / count
        matching_score_real = np.array([item['matching_score_real'] for item in results]).sum() / count
        R_precision = np.array([item['R_precision'] for item in results]).sum(axis=0) / count
        matching_score_pred = np.array([item['matching_score_pred'] for item in results]).sum() / count
        pred_motion_embd = np.concatenate([item['pred_motion_embd'].numpy() for item in results], axis=0)
        gt_motion_embd = np.concatenate([item['gt_motion_embd'].numpy() for item in results], axis=0)
        gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embd)
        mu, cov = calculate_activation_statistics(pred_motion_embd)

        diversity_real = calculate_diversity(gt_motion_embd, 300 if count > 300 else 100)
        diversity = calculate_diversity(pred_motion_embd, 300 if count > 300 else 100)

        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        return dict(
            R1_real=R_precision_real[0], 
            R2_real=R_precision_real[1], 
            R3_real=R_precision_real[2], 
            R1=R_precision[0],
            R2=R_precision[1],
            R3=R_precision[2], 
            fid=fid,
            div_real=diversity_real, 
            div=diversity, 
            matching_score_real=matching_score_real, 
            matching_score_pred=matching_score_pred
        )

@METRICS.register_module()
class Text2MotionTrajMetric(Text2MotionMetric):
    @torch.no_grad()
    def process(self, data_batch, data_samples):
        pred_motions = data_samples
        pred_motions = pad_sequence(pred_motions, batch_first=True)
        gt_motions = pad_sequence(data_batch['motion'], batch_first=True).cuda()
        motion_lengths = data_batch['motion_length']
        mean, std = data_batch['mean'].cuda(), data_batch['std'].cuda()
        captions = data_batch['text_data']

        pred_joints = recover_from_local_position((pred_motions * std + mean).cpu().numpy(), njoint=22)
        gt_joints = recover_from_local_position((gt_motions * std + mean).cpu().numpy(), njoint=22)

        # idx = 6;pred_data = recover_from_local_rotation((pred_motions * std + mean).cpu().numpy()[idx][:motion_lengths[idx]], njoint=22); np.savez('tmp.npz', **pred_data)
        # idx = 6;pred_data = recover_from_local_rotation((data_batch['motion'].cuda() * std + mean).cpu().numpy()[idx][:motion_lengths[idx]], njoint=22); np.savez('tmp.npz', **pred_data)
        # idx = 6;vis_motion(pred_motions[idx][:motion_lengths[idx]].cpu(), mean.cpu(), std.cpu())
        # idx = 6;vis_motion(data_batch['motion'][idx][:motion_lengths[idx]], mean.cpu(), std.cpu())
 
        pad_m = torch.zeros((pred_motions.shape[0], 300 - motion_lengths.max(), 272)).to(pred_motions)
        pred_motions = torch.cat([pred_motions, pad_m], dim=1)
        gt_motions = torch.cat([gt_motions, pad_m], dim=1)

        with torch.no_grad():
            gt_m_latents = self.tmr_model.motionencoder(gt_motions.cuda(), motion_lengths.cuda()).loc
            m_latents = self.tmr_model.motionencoder(pred_motions.cuda(), motion_lengths.cuda()).loc
            t_latents = self.tmr_model.textencoder(captions).loc

        R_precision_real, matching_score_real = calculate_R_precision(t_latents.cpu().numpy(), gt_m_latents.cpu().numpy(), top_k=3, sum_all=True)
        R_precision, matching_score_pred = calculate_R_precision(t_latents.cpu().numpy(), m_latents.cpu().numpy(), top_k=3, sum_all=True)


        # 算轨迹误差
        # (32, 300, 22, 3)
        B, T, J, C = pred_joints.shape
        mask = torch.arange(motion_lengths.max()).expand(gt_motions.shape[0], motion_lengths.max()) < motion_lengths[:, None].cpu() # B, T
        mask = mask[..., None, None].repeat(1, 1, J, C)
        mask[:, :, 1:] = 0
        mask = mask.permute(0, 2, 3, 1)
        # global_joint_mask = torch.zeros((B, J, 3, T), dtype=torch.float32)
        # global_joint_mask[:, 0, :, :] = 1
        kps_error, t_mask = compute_kps_error(
            torch.from_numpy(pred_joints).permute(0, 2, 3, 1), 
            torch.from_numpy(gt_joints).permute(0, 2, 3, 1), mask)
        err_np = calculate_trajectory_error(kps_error, t_mask)
        self.results.append({
            'R_precision_real': R_precision_real,
            'matching_score_real': matching_score_real, 
            'R_precision': R_precision, 
            'matching_score_pred': matching_score_pred, 
            # 'text_embd': text_embd, 
            'pred_motion_embd': m_latents, 
            'gt_motion_embd': gt_m_latents, 
            'traj_err': err_np,
            'samplt_cnt': len(captions)
        })

    def compute_metrics(self, results):
        count = sum(item['samplt_cnt'] for item in results)
        R_precision_real = np.array([item['R_precision_real'] for item in results]).sum(axis=0) / count
        matching_score_real = np.array([item['matching_score_real'] for item in results]).sum() / count
        R_precision = np.array([item['R_precision'] for item in results]).sum(axis=0) / count
        matching_score_pred = np.array([item['matching_score_pred'] for item in results]).sum() / count
        pred_motion_embd = np.concatenate([item['pred_motion_embd'].cpu().numpy() for item in results], axis=0)
        gt_motion_embd = np.concatenate([item['gt_motion_embd'].cpu().numpy() for item in results], axis=0)
        gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embd)
        mu, cov = calculate_activation_statistics(pred_motion_embd)

        diversity_real = calculate_diversity(gt_motion_embd, 300 if count > 300 else 100)
        diversity = calculate_diversity(pred_motion_embd, 300 if count > 300 else 100)

        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        traj_err = np.concatenate([item['traj_err'] for item in results], axis=0)
        traj_err = traj_err.reshape((-1, traj_err[0].shape[-1])).mean(0)
        return dict(
            R1_real=R_precision_real[0], 
            R2_real=R_precision_real[1], 
            R3_real=R_precision_real[2], 
            R1=R_precision[0],
            R2=R_precision[1],
            R3=R_precision[2], 
            fid=fid,
            div_real=diversity_real,  
            div=diversity, 
            matching_score_real=matching_score_real, 
            matching_score_pred=matching_score_pred, 
            traj_err=traj_err[1], 
            loc_err=traj_err[3], 
            avg_err=traj_err[4]
        )
