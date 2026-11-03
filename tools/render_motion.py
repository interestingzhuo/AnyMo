"""Render a generated AnyMo NPZ as a world-coordinate, Y-up skeleton MP4."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='NPZ containing joints (T, 22, 3) and fps')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--poster', type=Path, help='Optional PNG of the middle frame')
    parser.add_argument('--camera', choices=['fixed', 'follow'], default='fixed',
                        help='Fixed world bounds, or camera bounds following the pelvis')
    args = parser.parse_args(argv)
    if args.out.suffix.lower() != '.mp4':
        parser.error('--out must end in .mp4')
    if args.poster is not None and args.poster.suffix.lower() != '.png':
        parser.error('--poster must end in .png')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio_ffmpeg
    with np.load(args.input, allow_pickle=False) as sample:
        joints = np.asarray(sample['joints'], dtype=np.float32)
        fps = float(sample['fps'])
        target = np.asarray(sample['target_traj']) if 'target_traj' in sample else None
    if joints.ndim != 3 or joints.shape[1:] != (22, 3) or not len(joints) or not np.isfinite(joints).all():
        raise ValueError('joints must have finite shape (T, 22, 3)')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    if target is not None and (target.shape != (len(joints), 3) or not np.isfinite(target).all()):
        raise ValueError('target_traj must have finite shape (T, 3)')
    points = joints.reshape(-1, 3) if target is None else np.concatenate([joints.reshape(-1, 3), target])
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    radius = max(float((hi - lo).max()) / 2 + .2, .9)
    offsets = (joints - joints[:, :1]).reshape(-1, 3)
    body_lo, body_hi = offsets.min(0), offsets.max(0)
    follow_radius = max(float((body_hi - body_lo).max()) / 2 + .2, .9)
    fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[2]-radius, center[2]+radius),
           zlim=(center[1]-radius, center[1]+radius), xlabel='X (m)', ylabel='Z (m)', zlabel='Y (m)')
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=-65)
    chains = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
              [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    lines = [ax.plot([], [], [], linewidth=2.5)[0] for _ in chains]
    generated_path = ax.plot(joints[:, 0, 0], joints[:, 0, 2], joints[:, 0, 1],
                             color='gray', alpha=.5, label='Generated pelvis')[0]
    target_path = None
    if target is not None:
        target_path = ax.plot(target[:, 0], target[:, 2], target[:, 1], '--',
                              color='black', label='Target pelvis')[0]
    ax.legend(loc='upper left')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(args.out), (640, 640), fps=fps,
                                        codec='libx264', pix_fmt_out='yuv420p')
    try:
        writer.send(None)
        for i, pose in enumerate(joints):
            if args.camera == 'follow':
                center = pose[0] + (body_lo + body_hi)/2
                ax.set(xlim=(center[0]-follow_radius, center[0]+follow_radius),
                       ylim=(center[2]-follow_radius, center[2]+follow_radius),
                       zlim=(center[1]-follow_radius, center[1]+follow_radius))
                for path, line in [(joints[:, 0], generated_path), (target, target_path)]:
                    if path is None:
                        continue
                    visible = np.all(np.abs(path-center) <= follow_radius, axis=1)
                    segment = np.where(visible[:, None], path, np.nan)
                    line.set_data(segment[:, 0], segment[:, 2])
                    line.set_3d_properties(segment[:, 1])
            for chain, line in zip(chains, lines):
                line.set_data(pose[chain, 0], pose[chain, 2])
                line.set_3d_properties(pose[chain, 1])
            ax.set_title(f'AnyMo | {i + 1}/{len(joints)} | {fps:g} fps')
            fig.canvas.draw()
            if args.poster is not None and i == len(joints)//2:
                args.poster.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.poster)
            writer.send(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    finally:
        writer.close()
        plt.close(fig)
    print(args.out.resolve())


if __name__ == '__main__':
    main()
