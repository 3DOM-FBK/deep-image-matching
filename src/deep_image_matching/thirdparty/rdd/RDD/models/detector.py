import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from typing import Optional, Callable
from ..utils.misc import NestedTensor

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=False)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x

class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=False)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.gate(out)

        return out
    
class RDD_detector(nn.Module):
    def __init__(self, block_dims, hidden_dim=128):
        super().__init__()
        self.gate = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.block1 = ConvBlock(3, block_dims[0], self.gate, nn.BatchNorm2d)
        self.block2 = ResBlock(inplanes=block_dims[0], planes=block_dims[1], stride=1,
                            downsample=nn.Conv2d(block_dims[0], block_dims[1], 1),
                            gate=self.gate,
                            norm_layer=nn.BatchNorm2d)
        self.block3 = ResBlock(inplanes=block_dims[1], planes=block_dims[2], stride=1,
                            downsample=nn.Conv2d(block_dims[1], block_dims[2], 1),
                            gate=self.gate,
                            norm_layer=nn.BatchNorm2d)
        self.block4 = ResBlock(inplanes=block_dims[2], planes=block_dims[3], stride=1,
                            downsample=nn.Conv2d(block_dims[2], block_dims[3], 1),
                            gate=self.gate,
                            norm_layer=nn.BatchNorm2d)

        self.conv1 = resnet.conv1x1(block_dims[0], hidden_dim // 4)
        self.conv2 = resnet.conv1x1(block_dims[1], hidden_dim // 4)
        self.conv3 = resnet.conv1x1(block_dims[2], hidden_dim // 4)
        self.conv4 = resnet.conv1x1(block_dims[3], hidden_dim // 4)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.convhead2 = nn.Sequential(
                                resnet.conv1x1(hidden_dim, 1),
                                nn.Sigmoid()
        )

    def forward(self, samples: NestedTensor):
        x1 = self.block1(samples.tensors)
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool4(x2)
        x3 = self.block3(x3)  # B x c3 x H/8 x W/8
        x4 = self.pool4(x3)
        x4 = self.block4(x4)
        
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//8 x W//8
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//32 x W//32
        
        x2_up = self.upsample2(x2)
        x3_up = self.upsample8(x3)
        x4_up = self.upsample32(x4)
        
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)
        scoremap = self.convhead2(x1234)
        
        return scoremap
    
def build_detector(config):
    block_dims = config['block_dims']
    return RDD_detector(block_dims, block_dims[-1])