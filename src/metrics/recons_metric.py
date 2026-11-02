import torch
import numpy as np
from mmengine import Config
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS, MODELS
from scipy import linalg
from ..utils.rotation_conversions import rotation_6d_to_matrix

@METRICS.register_module()
class ReconsMetric(BaseMetric):
    def _compute_mpjpe(self, preds, target, valid_mask=None, pck_joints=None, sample_wise=True):
        """
        Mean per-joint position error (i.e. mean Euclidean distance)
        often referred to as "Protocol #1" in many papers.
        """
        assert preds.shape == target.shape, print(preds.shape, target.shape)  # BxJx3
        mpjpe = torch.norm(preds - target, p=2, dim=-1)  # BxJ

        if pck_joints is None:
            if sample_wise:
                mpjpe_seq = (
                    (mpjpe * valid_mask.float()).sum(-1) / valid_mask.float().sum(-1)
                    if valid_mask is not None
                    else mpjpe.mean(-1)
                )
            else:
                mpjpe_seq = mpjpe[valid_mask] if valid_mask is not None else mpjpe
            return mpjpe_seq
        else:
            mpjpe_pck_seq = mpjpe[:, pck_joints]
            return mpjpe_pck_seq

    def align_by_parts(self, joints, align_inds=None):
        if align_inds is None:
            return joints
        pelvis = joints[:, align_inds].mean(1)
        return joints - torch.unsqueeze(pelvis, dim=1)

    def calc_mpjpe(self, preds, target, align_inds=[0], sample_wise=True, trans=None):
        # Expects BxJx3
        valid_mask = target[:, :, 0] != -2.0
        # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
        if align_inds is not None:
            preds_aligned = self.align_by_parts(preds, align_inds=align_inds)
            if trans is not None:
                preds_aligned += trans
            target_aligned = self.align_by_parts(target, align_inds=align_inds)
        else:
            preds_aligned, target_aligned = preds, target
        mpjpe_each = self._compute_mpjpe(
            preds_aligned, target_aligned, valid_mask=valid_mask, sample_wise=sample_wise
        )
        return mpjpe_each

    def batch_compute_similarity_transform_torch(self, S1, S2):
        """
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        """
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0, 2, 1)
            S2 = S2.permute(0, 2, 1)
            transposed = True
        assert S2.shape[1] == S1.shape[1]

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0, 2, 1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0], 1, 1)
        Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0, 2, 1)

        return S1_hat, (scale, R, t)

    def calc_pampjpe(self, preds, target, sample_wise=True, return_transform_mat=False):
        # Expects BxJx3
        target, preds = target.float(), preds.float()

        preds_tranformed, PA_transform = self.batch_compute_similarity_transform_torch(
            preds, target
        )
        pa_mpjpe_each = self._compute_mpjpe(preds_tranformed, target, sample_wise=sample_wise)

        # if return_transform_mat:
        #     return pa_mpjpe_each, PA_transform
        # else:
        return pa_mpjpe_each

    def calc_accel(self, preds, target):
        """
        Mean joint acceleration error
        often referred to as "Protocol #1" in many papers.
        """
        assert preds.shape == target.shape, print(preds.shape, target.shape)  # BxJx3
        assert preds.dim() == 3
        # Expects BxJx3
        # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
        accel_gt = target[:-2] - 2 * target[1:-1] + target[2:]
        accel_pred = preds[:-2] - 2 * preds[1:-1] + preds[2:]
        normed = torch.linalg.norm(accel_pred - accel_gt, dim=-1)
        accel_seq = normed.mean(1)
        return accel_seq

    @torch.no_grad()
    def process(self, data_batch, data_samples):
        pred_motions, indices = data_samples
        gt_motions = data_batch['motion'].cuda()
        motion_lengths = data_batch['motion_length'].cuda()
        mean, std = data_batch['mean'].cuda(), data_batch['std'].cuda()
        max_len = gt_motions.shape[1]
        mask = torch.arange(max_len, device=gt_motions.device).expand(
            motion_lengths.shape[0], max_len) < motion_lengths.unsqueeze(1)

        mpjpe, pampjpe, accel, w_mpjpe = [], [], [], []
        for i in range(len(pred_motions)):
            T = motion_lengths[i]
            pred_motion = pred_motions[i][:T].squeeze() * std[None] + mean[None]
            gt_motion = gt_motions[i, :T] * std + mean

            pred_joints = recover_from_local_position(pred_motion.cpu().numpy(), njoint=22)
            gt_joints = recover_from_local_position(gt_motion.cpu().numpy(), njoint=22)
            pred_joints = torch.from_numpy(pred_joints).float()
            gt_joints = torch.from_numpy(gt_joints).float()

            # from ..utils.visualize import vis_3d_motion
            # vis_3d_motion(pred_joints[None].numpy(), None, [f'video.mp4'], fps=30)
            # vis_3d_motion(gt_joints[None].numpy(), None, [f'gt_video.mp4'], fps=30)
        
            pampjpe.append(self.calc_pampjpe(gt_joints * 1000, pred_joints * 1000).mean())
            accel.append(self.calc_accel(gt_joints * 1000, pred_joints * 1000).mean())
            mpjpe.append(self.calc_mpjpe(gt_joints * 1000, pred_joints * 1000).mean())
            w_mpjpe.append(self._compute_mpjpe(pred_joints - pred_joints[0, 0], gt_joints - gt_joints[0, 0]).mean() * 1000)

        self.results.append({
            'mpjpe': mpjpe,
            'pampjpe': pampjpe, 
            'w_mpjpe': w_mpjpe, 
            'accel': accel, 
            'count': len(mpjpe), 
        })

    def compute_metrics(self, results):
        count = sum(item['count'] for item in results)
        mpjpe = sum([i for item in results for i in item['mpjpe']])
        pampjpe = sum([i for item in results for i in item['pampjpe']])
        w_mpjpe = sum([i for item in results for i in item['w_mpjpe']])
        accel = sum([i for item in results for i in item['accel']])

        return dict(
            mpjpe=mpjpe / count,
            pampjpe=pampjpe / count, 
            accel=accel / count,
            w_mpjpe=w_mpjpe / count, 
        )

def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

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


def accumulate_rotations(relative_rotations):
    R_total = [relative_rotations[0]]
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    return np.array(R_total)

def recover_from_local_position(final_x, njoint):

    if final_x.ndim == 3:
        bs, nfrm, _ = final_x.shape
        is_batched = True
    else:
        nfrm, _ = final_x.shape
        bs = 1
        is_batched = False
        final_x = final_x.reshape(1, *final_x.shape)

    
    positions_no_heading = final_x[:,:,8:8+3*njoint].reshape(bs, nfrm, njoint, 3) 
    velocities_root_xy_no_heading = final_x[:,:,:2] 
    global_heading_diff_rot = final_x[:,:,2:8] 

   
    positions_with_heading = []
    for b in range(bs):
        
        global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot[b])).numpy())
        inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
        
        
        curr_pos_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:, None,:, :], njoint, axis=1), 
                                        positions_no_heading[b][...,None]).squeeze(-1)

        
        velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading[b].shape[0], 3))
        velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[b, :, 0]
        velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[b, :, 1]
        velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], 
                                                         velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)

        root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
        
        curr_pos_with_heading[:, :, 0] += root_translation[:, 0:1]
        curr_pos_with_heading[:, :, 2] += root_translation[:, 2:]
        
        positions_with_heading.append(curr_pos_with_heading)

    positions_with_heading = np.stack(positions_with_heading, axis=0)

    if not is_batched:
        positions_with_heading = positions_with_heading.squeeze(0)

    return positions_with_heading
