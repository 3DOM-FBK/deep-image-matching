import torch
import torch.nn as nn


class BackboneBase(nn.Module):
    """Base class for backbone networks. Provides a standard interface for preprocessing inputs and
    defining encoder dimensions.

    Args:
        nchannels (int): Number of input channels.
        use_instance_norm (bool): Whether to apply instance normalization.
    """

    def __init__(self, nchannels=3, use_instance_norm=False):
        super().__init__()
        assert nchannels > 0, "Number of channels must be positive."
        self.nchannels = nchannels
        self.use_instance_norm = use_instance_norm
        self.norm = nn.InstanceNorm2d(nchannels) if use_instance_norm else None

    def get_dim_layers_encoder(self):
        """Get dimensions of encoder layers."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _forward(self, x):
        """Define the forward pass for the backbone."""
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self, x: torch.Tensor, preprocess=True):
        """Forward pass with optional preprocessing.

        Args:
            x (Tensor): Input tensor.
            preprocess (bool): Whether to apply channel reduction.
        """
        if preprocess:
            if x.dim() != 4:
                if x.dim() == 2 and x.shape[0] > 3 and x.shape[1] > 3:
                    x = x.unsqueeze(0).unsqueeze(0)
                elif x.dim() == 3:
                    x = x.unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected input shape: {x.shape}")

            if self.nchannels == 1 and x.shape[1] != 1:
                if len(x.shape) == 4:  # Assumes (batch, channel, height, width)
                    x = torch.mean(x, axis=1, keepdim=True)
                else:
                    raise ValueError(f"Unexpected input shape: {x.shape}")

            #
            if self.nchannels == 3 and x.shape[1] == 1:
                if len(x.shape) == 4:
                    x = x.repeat(1, 3, 1, 1)
                else:
                    raise ValueError(f"Unexpected input shape: {x.shape}")

        if self.use_instance_norm:
            x = self.norm(x)

        return self._forward(x)
