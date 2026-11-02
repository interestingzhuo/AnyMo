"""Portable inference helpers for full released AnyMo checkpoints."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import MODELS
from .utils.motion import recover_from_local_position_fast


def load_model(checkpoint, config, assets_dir, condition_types):
    from . import models  # Register model components without evaluation packages.
    assets_dir = Path(assets_dir)
    cfg = Config.fromfile(str(config))
    cfg.model.init_cfg = None
    cfg.model.vqvae_cfg.init_cfg = None
    cfg.model.stage = 4
    cfg.model.cond_types = condition_types
    cfg.model.text_encoder_cfg.update(checkpoint_root=str(assets_dir / 'checkpoints'),
                                     pretrained=False, local_files_only=True)
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    state = saved.get('state_dict', saved)
    if all(key.startswith('module.') for key in state):
        state = {key.removeprefix('module.'): value for key, value in state.items()}
    for prefix in ['text_encoder.text_encoder.encoder.block.', 'vqvae.encoder.']:
        if not any(key.startswith(prefix) for key in state):
            raise ValueError(f'Expected a full released checkpoint containing {prefix}')
    model = MODELS.build(cfg.model)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).cuda()
    return model, cfg


def load_stats(assets_dir, with_trajectory=False):
    names = ['mean_272', 'std_272'] + (['traj_mean', 'traj_std'] if with_trajectory else [])
    stats = {}
    for name in names:
        path = Path(assets_dir) / 'data' / (name + '.npy')
        value = np.load(path, allow_pickle=False).astype(np.float32)
        expected = (3,) if name.startswith('traj_') else (272,)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f'{path} must be finite with shape {expected}')
        if 'std' in name and (value < 0).any():
            raise ValueError(f'{path} contains negative standard deviations')
        if name == 'traj_std' and (value <= 0).any():
            raise ValueError('Trajectory standard deviations must be positive')
        stats[name] = torch.from_numpy(value).cuda()
    return stats


def read_trajectory(path):
    """Read unnormalized pelvis XYZ positions in meters at 30fps."""
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        with value:
            if 'target_traj' not in value:
                raise ValueError('A trajectory NPZ must contain target_traj with shape (T, 3)')
            trajectory = value['target_traj'].astype(np.float32)
    else:
        trajectory = value.astype(np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or not np.isfinite(trajectory).all():
        raise ValueError('Trajectory must be a finite (T, 3) array')
    return trajectory


def fit_trajectory(model, tokens, target, mean, std, steps=400, lr=2e-4):
    """Optimize decoder latents while preserving model weights and sampled tokens."""
    mask = (tokens > -1).unsqueeze(1)
    latent_lengths = mask[0].squeeze(0).sum(1)
    with torch.no_grad():
        latent = model.vqvae.quantizer.dequantize(tokens.clamp_min(0).permute(1, 2, 0))
        latent = latent.permute(0, 2, 1) * mask[0][0].unsqueeze(1)
    latent = latent.detach().clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([latent], lr=lr, betas=(.5, .9), weight_decay=1e-6)
    losses = []
    for step in range(steps):
        motion = model.vqvae.decoder(latent, latent_lengths).permute(0, 2, 1)[:, :len(target)]
        pelvis = recover_from_local_position_fast(motion * std + mean, 22)[0, :, 0]
        loss = F.mse_loss(pelvis, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite trajectory loss at iteration {step}; native output remains saved')
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(latent.grad).all():
            raise RuntimeError(f'Nonfinite latent gradient at iteration {step}; native output remains saved')
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step == steps - 1:
            losses.append(dict(step=step + 1, loss=float(loss.detach())))
    with torch.no_grad():
        motion = model.vqvae.decoder(latent, latent_lengths).permute(0, 2, 1)[:, :len(target)]
    if not torch.isfinite(motion).all():
        raise RuntimeError('Nonfinite optimized motion; native output remains saved')
    return motion, losses


def save_motion(path, normalized, stats, tokens, prompt, trajectory=None):
    with torch.no_grad():
        motion = normalized * stats['std_272'] + stats['mean_272']
        joints = recover_from_local_position_fast(motion, 22)[0].cpu().numpy()
        values = dict(normalized=normalized[0].cpu().numpy(), motion=motion[0].cpu().numpy(),
                      joints=joints, tokens=tokens.cpu().numpy(), prompt=prompt or '', fps=30)
        if trajectory is not None:
            values['target_traj'] = trajectory.cpu().numpy()
        if not np.isfinite(joints).all():
            raise RuntimeError('Nonfinite recovered joints')
        np.savez_compressed(path, **values)
        result = dict(root_path_m=float(np.linalg.norm(np.diff(joints[:, 0], axis=0), axis=-1).sum()))
        if trajectory is not None:
            errors = np.linalg.norm(joints[:, 0] - values['target_traj'], axis=-1)
            result.update(ade_m=float(errors.mean()), fde_m=float(errors[-1]))
        return result
