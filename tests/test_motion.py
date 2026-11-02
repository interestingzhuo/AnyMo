import unittest
import numpy as np
import torch
from src.utils.motion import recover_from_local_position_fast, recover_from_local_position
from src.utils.rotation_conversions import rotation_6d_to_matrix
from src.utils.motion import accumulate_rotations_batch


def legacy_recovery(x):
    """Reference implementation for coordinate and autograd regression tests."""
    b,t,_=x.shape
    inverse=accumulate_rotations_batch(rotation_6d_to_matrix(x[:,:,2:8])).transpose(-1,-2)
    pos=(inverse[:,:,None] @ x[:,:,8:74].reshape(b,t,22,3,1)).squeeze(-1)
    velocity=torch.zeros(b,t,3,device=x.device)
    velocity[:,:,0]=x[:,:,0];velocity[:,:,2]=x[:,:,1]
    velocity[:,1:]=(inverse[:,:-1] @ velocity[:,1:,...,None]).squeeze(-1)
    translation=velocity.cumsum(1)
    pos[:,:,:,0]+=translation[:,:,0:1];pos[:,:,:,2]+=translation[:,:,2:3]
    return pos


def sample(frames=8,dtype=torch.float32):
    torch.manual_seed(123)
    x=torch.randn(1,frames,272,dtype=dtype)*.01
    x[:,:,2:8]+=torch.tensor([1,0,0,0,1,0],dtype=dtype)
    return x


class MotionTests(unittest.TestCase):
    def test_forward_matches_original_and_numpy(self):
        for frames in [1,8,60]:
            x=sample(frames)
            torch.testing.assert_close(recover_from_local_position_fast(x),legacy_recovery(x),rtol=1e-6,atol=1e-6)
            expected=recover_from_local_position(x.numpy()[0],22)
            np.testing.assert_allclose(recover_from_local_position_fast(x).numpy()[0],expected,rtol=1e-5,atol=1e-6)

    def test_original_backward_failure_and_fixed_gradient(self):
        x=sample().requires_grad_(True)
        with self.assertRaises(RuntimeError):legacy_recovery(x).square().mean().backward()
        x=sample(dtype=torch.float64).requires_grad_(True)
        direction=torch.randn_like(x)*.01
        loss=recover_from_local_position_fast(x).square().mean()
        gradient=torch.autograd.grad(loss,x)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        eps=1e-5
        numerical=(recover_from_local_position_fast(x.detach()+eps*direction).square().mean()-
                   recover_from_local_position_fast(x.detach()-eps*direction).square().mean())/(2*eps)
        self.assertAlmostEqual(float((gradient*direction).sum()),float(numerical),places=7)


if __name__=='__main__':unittest.main()
