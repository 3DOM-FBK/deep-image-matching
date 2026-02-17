import os
import sys
import numpy as np

featureboost_config = {
    "keypoint_dim": 65,
    "keypoint_encoder": [128, 64, 64],
    "normal_dim": 192,
    "normal_encoder": [128, 64, 64],
    "descriptor_encoder": [64, 64],
    "descriptor_dim": 64,
    "Attentional_layers": 3,
    "last_activation": None,
    "l2_normalization": None,
    "output_dim": 64,
}