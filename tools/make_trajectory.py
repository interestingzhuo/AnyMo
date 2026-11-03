"""Create an analytic figure-eight pelvis trajectory; no dataset assets required."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('examples/figure_eight.npy'))
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--height', type=float, default=.95)
    parser.add_argument('--radius', type=float, default=.6)
    args = parser.parse_args()
    if args.out.suffix.lower() != '.npy':
        parser.error('--out must end in .npy')
    import numpy as np
    if not 60 <= args.frames <= 300 or args.frames % 4:
        parser.error('frames must be a multiple of four between 60 and 300')
    if not np.isfinite([args.height, args.radius]).all() or min(args.height, args.radius) <= 0:
        parser.error('height and radius must be finite and positive')
    t = np.linspace(0, 2*np.pi, args.frames, dtype=np.float32)
    trajectory = np.stack([args.radius*np.sin(t), np.full_like(t, args.height),
                           .5*args.radius*np.sin(2*t)], axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, trajectory)
    print(args.out.resolve())


if __name__ == '__main__':
    main()
