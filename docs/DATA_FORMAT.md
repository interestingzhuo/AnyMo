# Motion and trajectory conventions

The public inference entry point uses the existing 272-dimensional representation and 22-joint position recovery. All documented outputs are 30fps. A clip must contain 60–300 frames and its length must be divisible by 4.

The recovery routine uses root planar velocities from channels 0:2, relative 6D heading rotations from 2:8, and heading-free joint XYZ positions from 8:74. It accumulates relative heading rotations, transforms local joints, then integrates world planar root velocity. Y is the vertical axis. Do not interpret the full 272 channels as SMPL parameters or a BVH file.

`motion = normalized * std_272 + mean_272`. For trajectory conditioning, `traj_input = (target_xyz - traj_mean) / traj_std`. The trajectory is the pelvis/root joint (`joints[:,0]`) in meters. A zero Y coordinate places the pelvis at floor height, not at a standing person's hip height.

Use a NumPy array `(T,3)` in `.npy`, or an `.npz` with `target_traj`. Input paths are not recentered or rotated; preserve the original coordinate convention. The endpoint error is the final XYZ Euclidean distance; ADE is the mean per-frame XYZ distance. Neither measure aligns or rescales the prediction.

Training pickle files contain lists of records such as `motion_272`, `captions`, and optionally `audio_feat`. Training pickle files must come from a trusted source. Prepared test data is a subset and is not the complete training corpus. Speech/music input is precomputed audio features, not raw waveform input to the documented CLI.

Split lists live in `configs/data_paths/`; data paths inside the lists are relative to the repository root. Source dataset preparation is documented at https://huggingface.co/datasets/L-yiheng/OmniHuMo .
