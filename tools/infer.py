"""Generate motion from text, a pelvis trajectory, or both."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None, require_trajectory=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/inference/anymo_3b.py')
    parser.add_argument('--assets-dir', type=Path, default=ROOT,
                        help='Directory containing data/*.npy and checkpoints/flan-t5-xl/')
    parser.add_argument('--prompt')
    parser.add_argument('--trajectory', type=Path, required=require_trajectory)
    parser.add_argument('--frames', type=int, help='Default: trajectory length, or 300 for text only')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--sampling-steps', type=int, default=6)
    parser.add_argument('--topk-threshold', type=float, default=.9,
                        help='0.9 retains the highest-scoring 10 percent; this is not top-p')
    parser.add_argument('--tta-steps', type=int, default=400 if require_trajectory else 0)
    parser.add_argument('--tta-lr', type=float, default=2e-4)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/sample')
    args = parser.parse_args(argv)
    if not args.prompt and args.trajectory is None:
        parser.error('Provide --prompt, --trajectory, or both')
    if args.sampling_steps < 1 or not 0 <= args.topk_threshold < 1:
        parser.error('sampling-steps must be positive; topk-threshold must lie in [0, 1)')
    if args.tta_steps < 0 or args.tta_lr <= 0 or (args.tta_steps and args.trajectory is None):
        parser.error('TTA requires a trajectory, nonnegative steps, and a positive learning rate')
    if args.frames is not None and not (60 <= args.frames <= 300 and args.frames % 4 == 0):
        parser.error('frames must be a multiple of 4 between 60 and 300')
    return args


def main(argv=None, require_trajectory=False):
    args = parse_args(argv, require_trajectory)
    # Help and argument validation do not require GPU dependencies.
    import torch
    from src.inference import load_model, load_stats, read_trajectory, fit_trajectory, save_motion
    if not torch.cuda.is_available():
        raise RuntimeError('This release uses CUDA for the text encoder and motion model')
    trajectory = read_trajectory(args.trajectory) if args.trajectory is not None else None
    frames = args.frames or (len(trajectory) if trajectory is not None else 300)
    if not (60 <= frames <= 300 and frames % 4 == 0):
        raise ValueError('Use 60..300 frames in multiples of 4; trajectories are not silently resampled')
    if trajectory is not None and len(trajectory) != frames:
        raise ValueError('frames must equal the trajectory length')
    condition_types = (['text'] if args.prompt else []) + (['traj'] if trajectory is not None else [])
    stats = load_stats(args.assets_dir.resolve(), trajectory is not None)
    model, cfg = load_model(args.checkpoint.resolve(), args.config.resolve(), args.assets_dir.resolve(), condition_types)
    model.inference_timesteps = args.sampling_steps
    model.topk_filter_thres = args.topk_threshold
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    inputs = dict(motion=torch.zeros(1, frames, 272, device='cuda'),
                  motion_length=torch.tensor([frames], device='cuda'), text_data=[args.prompt or ''],
                  mode='predict', return_ids=True)
    target = None
    if trajectory is not None:
        target = torch.from_numpy(trajectory).cuda()
        inputs['traj_data'] = ((target - stats['traj_mean']) / stats['traj_std'])[None]
    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize();started = time.monotonic()
    with torch.no_grad():
        normalized, _, tokens = model(**inputs)
        if isinstance(normalized, list):
            normalized = torch.stack(normalized)
        normalized = normalized[:, :frames].float()
    torch.cuda.synchronize()
    if normalized.shape != (1, frames, 272) or not torch.isfinite(normalized).all():
        raise RuntimeError('Invalid generated motion')
    report = dict(status='native_complete', seed=args.seed, frames=frames, fps=30, precision='float32',
                  checkpoint=str(args.checkpoint.resolve()), strict_load=True, conditions=condition_types,
                  sampling_steps=args.sampling_steps, topk_threshold=args.topk_threshold,
                  inference_seconds=time.monotonic() - started, torch=torch.__version__, cuda=torch.version.cuda)
    report['gpu'] = torch.cuda.get_device_name()
    report['native'] = save_motion(args.out / 'native.npz', normalized, stats, tokens, args.prompt, target)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    if args.tta_steps:
        try:
            optimized, losses = fit_trajectory(model, tokens, target, stats['mean_272'], stats['std_272'],
                                               args.tta_steps, args.tta_lr)
        except RuntimeError as error:
            report.update(status='tta_failed', error=str(error))
            (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            raise
        report['optimized'] = save_motion(args.out / 'optimized.npz', optimized, stats, tokens, args.prompt, target)
        report['tta'] = dict(steps=args.tta_steps, lr=args.tta_lr, losses=losses)
    report['status'] = 'complete'
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    cfg.dump(str(args.out / 'resolved_config.py'))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
