import torch
import torch.nn as nn
from mmengine.registry import MODELS

@MODELS.register_module()
class ReconsLoss(nn.Module):
    def __init__(self, lambda_recons=1.0, lambda_pos=0.5):
        super().__init__()
        self.lambda_recons = lambda_recons
        self.lambda_pos = lambda_pos
        self.l1_smooth_loss = torch.nn.SmoothL1Loss(reduction='none')

    def forward(self, pred_motion, gt_motion, motion_length, commit_loss=None):
        mask = length_to_mask(motion_length, max_len=pred_motion.shape[1])
        
        loss_rec = self.l1_smooth_loss(pred_motion, gt_motion)
        loss_rec = (loss_rec * mask).sum() / (mask.sum() * pred_motion.shape[-1])

        loss_pos = self.l1_smooth_loss(pred_motion[..., 8:8+22*3], gt_motion[..., 8:8+22*3])
        loss_pos = (loss_pos * mask).sum() / (mask.sum() * loss_pos.shape[-1])

        return dict(
            recons_loss=loss_rec * self.lambda_recons,
            pos_loss=loss_pos * self.lambda_pos, 
            commit_loss=0.0 if commit_loss is None else commit_loss * 0.02
        )

def length_to_mask(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max()
    arange_t = torch.arange(max_len, device=lengths.device)[None, :]  # (1, T)
    return (arange_t < lengths[:, None]).unsqueeze(-1)  # (B, T, 1)
