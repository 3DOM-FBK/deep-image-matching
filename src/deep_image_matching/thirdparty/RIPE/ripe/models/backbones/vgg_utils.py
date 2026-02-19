# adapted from: https://github.com/Parskatt/DeDoDe/blob/main/DeDoDe/encoder.py and https://github.com/Parskatt/DeDoDe/blob/main/DeDoDe/decoder.py

import torch
import torch.nn as nn
import torchvision.models as tvm

from ... import utils

log = utils.get_pylogger(__name__)


class Decoder(nn.Module):
    def __init__(self, layers, *args, super_resolution=False, num_prototypes=1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.scales = self.layers.keys()
        self.super_resolution = super_resolution
        self.num_prototypes = num_prototypes

    def forward(self, features, context=None, scale=None):
        if context is not None:
            features = torch.cat((features, context), dim=1)
        stuff = self.layers[scale](features)
        logits, context = (
            stuff[:, : self.num_prototypes],
            stuff[:, self.num_prototypes :],
        )
        return logits, context


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=True,
        kernel_size=5,
        hidden_blocks=5,
        residual=False,
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim,
            hidden_dim,
            dw=False,
            kernel_size=1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.residual = residual

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=True,
        kernel_size=5,
        bias=True,
        norm_type=nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert out_dim % in_dim == 0, "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = norm_type(out_dim) if norm_type is nn.BatchNorm2d else norm_type(num_channels=out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, feats):
        b, c, hs, ws = feats.shape
        x0 = self.block1(feats)
        x = self.hidden_blocks(x0)
        if self.residual:
            x = (x + x0) / 1.4
        x = self.out_conv(x)
        return x


class VGG19(nn.Module):
    def __init__(self, pretrained=False, num_input_channels=3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        # Maxpool layers: 6, 13, 26, 39

        if num_input_channels != 3:
            log.info(f"Changing input channels from 3 to {num_input_channels}")
            self.layers[0] = nn.Conv2d(num_input_channels, 64, 3, 1, 1)

    def get_dim_layers(self):
        return [64, 128, 256, 512]

    def forward(self, x, **kwargs):
        feats = []
        sizes = []
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feats.append(x)
                sizes.append(x.shape[-2:])
            x = layer(x)
        return feats, sizes


class VGG(nn.Module):
    def __init__(self, size="19", pretrained=False) -> None:
        super().__init__()
        if size == "11":
            self.layers = nn.ModuleList(tvm.vgg11_bn(pretrained=pretrained).features[:22])
        elif size == "13":
            self.layers = nn.ModuleList(tvm.vgg13_bn(pretrained=pretrained).features[:28])
        elif size == "19":
            self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        # Maxpool layers: 6, 13, 26, 39

    def forward(self, x, **kwargs):
        feats = []
        sizes = []
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feats.append(x)
                sizes.append(x.shape[-2:])
            x = layer(x)
        return feats, sizes
