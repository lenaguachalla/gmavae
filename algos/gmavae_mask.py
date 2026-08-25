from algos.gmavae import GMAVAE
import torch

class GMAVAEMask(GMAVAE):
    """GMA-VAE with a row-wise (instead of block-diagonal) pi-mask."""

    def encode_action(self, A):
        Az = self.action_encoder(A)

        if self.group_masking:
            batch_size = Az.shape[:-2]

            if self.continuous_action_encoder:
                pi = self.pi[A[..., 0].long()]
            else:
                pi = self.pi[self.groups[A]]

            # duplicate pi across columns
            ones = torch.ones_like(pi)
            Pi = torch.einsum('...i,...j->...ij', pi, ones)

            I = torch.eye(self.z_dim).to(self.device)
            I = I.reshape(tuple(1 for _ in range(len(batch_size))) + (self.z_dim, self.z_dim))
            I = I.repeat(*(batch_size + (1, 1)))

            Az = Pi * (Az - I) + I

        return Az