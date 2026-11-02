# Example inputs

Use a specific text prompt from the root README, or supply a pelvis trajectory from your own motion data. The code candidate does not silently redistribute the reference dataset sequence used during validation.

To extract a trajectory from an already prepared inference output:

```python
import numpy as np
sample = np.load('path/to/motion.npz', allow_pickle=False)
np.save('trajectory.npy', sample['joints'][:, 0].astype(np.float32))
```

Use the source frame rate and coordinate convention. For the released inference path, the trajectory must be at30fps, contain60–300frames, and have a frame count divisible by4. A path with constant zero pelvis height is not a standing trajectory.
