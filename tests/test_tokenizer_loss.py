import unittest
import torch
from torch import nn
from src.models.tokenizers.vqvae import VQVAE
from src.losses.recons_loss import ReconsLoss

class Downsample(nn.Module):
    def forward(self, x, motion_length):
        return x[:, :, ::4] * 0

class Quantizer(nn.Module):
    def forward(self, x, motion_length):
        return x, x.sum() * 0, None, None, None

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.))
    def forward(self, x, motion_length):
        return x.repeat_interleave(4, dim=-1) + self.bias

class TokenizerLossTests(unittest.TestCase):
    def test_reconstruction_includes_late_valid_frames(self):
        model = object.__new__(VQVAE)
        nn.Module.__init__(model)
        model.encoder = Downsample()
        model.quantizer = Quantizer()
        model.decoder = Decoder()
        model.recons_loss = ReconsLoss()
        motion = torch.zeros(1, 8, 272)
        motion[:, 4:] = 1
        losses = model.forward_loss(motion, torch.tensor([8]))
        # Half the valid frames differ by one; SmoothL1(1) is 0.5.
        self.assertAlmostEqual(losses['recons_loss'].item(), 0.25)
        sum(losses.values()).backward()
        self.assertLess(model.decoder.bias.grad.item(), 0.)
        self.assertTrue(torch.isfinite(model.decoder.bias.grad))

if __name__ == '__main__':
    unittest.main()
