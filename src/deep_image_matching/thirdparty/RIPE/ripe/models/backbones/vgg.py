# adapted from: https://github.com/Parskatt/DeDoDe/blob/main/DeDoDe/encoder.py and https://github.com/Parskatt/DeDoDe/blob/main/DeDoDe/decoder.py

import torch.nn as nn
import torch.nn.functional as F

from .backbone_base import BackboneBase
from .vgg_utils import VGG19, ConvRefiner, Decoder


class VGG(BackboneBase):
    def __init__(self, nchannels=3, pretrained=True, use_instance_norm=True, mode="dect"):
        super().__init__(nchannels=nchannels, use_instance_norm=use_instance_norm)

        self.nchannels = nchannels
        self.mode = mode

        if self.mode not in ["dect", "desc", "dect+desc"]:
            raise ValueError("mode should be 'dect', 'desc' or 'dect+desc'")

        NUM_OUTPUT_CHANNELS, hidden_blocks = self._get_mode_params(mode)
        conv_refiner = self._create_conv_refiner(NUM_OUTPUT_CHANNELS, hidden_blocks)

        self.encoder = VGG19(pretrained=pretrained, num_input_channels=nchannels)
        self.decoder = Decoder(conv_refiner, num_prototypes=NUM_OUTPUT_CHANNELS)

    def _get_mode_params(self, mode):
        """Get the number of output channels and the number of hidden blocks for the ConvRefiner.

        Depending on the mode, the ConvRefiner will have a different number of output channels.
        """

        if mode == "dect":
            return 1, 8
        elif mode == "desc":
            return 256, 5
        elif mode == "dect+desc":
            return 256 + 1, 8

    def _create_conv_refiner(self, num_output_channels, hidden_blocks):
        return nn.ModuleDict(
            {
                "8": ConvRefiner(
                    512,
                    512,
                    256 + num_output_channels,
                    hidden_blocks=hidden_blocks,
                    residual=True,
                ),
                "4": ConvRefiner(
                    256 + 256,
                    256,
                    128 + num_output_channels,
                    hidden_blocks=hidden_blocks,
                    residual=True,
                ),
                "2": ConvRefiner(
                    128 + 128,
                    128,
                    64 + num_output_channels,
                    hidden_blocks=hidden_blocks,
                    residual=True,
                ),
                "1": ConvRefiner(
                    64 + 64,
                    64,
                    1 + num_output_channels,
                    hidden_blocks=hidden_blocks,
                    residual=True,
                ),
            }
        )

    def get_dim_layers_encoder(self):
        return self.encoder.get_dim_layers()

    def _forward(self, x):
        features, sizes = self.encoder(x)
        output = 0
        context = None
        scales = self.decoder.scales
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_descriptor, context = self.decoder(feature_map, scale=scale, context=context)
            output = output + delta_descriptor
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                output = F.interpolate(output, size=size, mode="bilinear", align_corners=False)
                context = F.interpolate(context, size=size, mode="bilinear", align_corners=False)

        if self.mode == "dect":
            return {"heatmap": output, "coarse_descs": features}
        elif self.mode == "desc":
            return {"fine_descs": output, "coarse_descs": features}
        elif self.mode == "dect+desc":
            logits = output[:, :1].contiguous()
            descs = output[:, 1:].contiguous()

            return {"heatmap": logits, "fine_descs": descs, "coarse_descs": features}
        else:
            raise ValueError("mode should be 'dect', 'desc' or 'dect+desc'")
