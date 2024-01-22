import torch
import torch.nn as nn
import torch.nn.functional as F

from ...DeDoDe.utils import get_grid
from .layers.block import Block
from .layers.attention import MemEffAttention
from .dinov2 import vit_large
