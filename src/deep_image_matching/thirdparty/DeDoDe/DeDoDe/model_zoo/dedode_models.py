import torch
import torch.nn as nn

from ...DeDoDe.detectors.dedode_detector import DeDoDeDetector
from ...DeDoDe.descriptors.dedode_descriptor import DeDoDeDescriptor
from ...DeDoDe.decoder import ConvRefiner, Decoder
from ...DeDoDe.encoder import VGG19, VGG, VGG_DINOv2
from ...DeDoDe.utils import get_best_device


def dedode_detector_B(device=get_best_device(), weights=None):
    residual = True
    hidden_blocks = 5
    amp_dtype = torch.float16
    amp = True
    NUM_PROTOTYPES = 1
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    encoder = VGG19(pretrained=False, amp=amp, amp_dtype=amp_dtype)
    decoder = Decoder(conv_refiner)
    model = DeDoDeDetector(encoder=encoder, decoder=decoder).to(device)
    if weights is not None:
        model.load_state_dict(weights)
    return model


def dedode_detector_L(device=get_best_device(), weights=None):
    NUM_PROTOTYPES = 1
    residual = True
    hidden_blocks = 8
    amp_dtype = (
        torch.float16
    )  # torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                128,
                64 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 64,
                64,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    encoder = VGG19(pretrained=False, amp=amp, amp_dtype=amp_dtype)
    decoder = Decoder(conv_refiner)
    model = DeDoDeDetector(encoder=encoder, decoder=decoder).to(device)
    if weights is not None:
        model.load_state_dict(weights)
    return model


def dedode_descriptor_B(device=get_best_device(), weights=None):
    NUM_PROTOTYPES = 256  # == descriptor size
    residual = True
    hidden_blocks = 5
    amp_dtype = (
        torch.float16
    )  # torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    encoder = VGG(size="19", pretrained=False, amp=amp, amp_dtype=amp_dtype)
    decoder = Decoder(conv_refiner, num_prototypes=NUM_PROTOTYPES)
    model = DeDoDeDescriptor(encoder=encoder, decoder=decoder).to(device)
    if weights is not None:
        model.load_state_dict(weights)
    return model


def dedode_descriptor_G(device=get_best_device(), weights=None, dinov2_weights=None):
    NUM_PROTOTYPES = 256  # == descriptor size
    residual = True
    hidden_blocks = 5
    amp_dtype = (
        torch.float16
    )  # torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "14": ConvRefiner(
                1024,
                768,
                512 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "8": ConvRefiner(
                512 + 512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    vgg_kwargs = dict(size="19", pretrained=False, amp=amp, amp_dtype=amp_dtype)
    dinov2_kwargs = dict(amp=amp, amp_dtype=amp_dtype, dinov2_weights=dinov2_weights)
    encoder = VGG_DINOv2(vgg_kwargs=vgg_kwargs, dinov2_kwargs=dinov2_kwargs)
    decoder = Decoder(conv_refiner, num_prototypes=NUM_PROTOTYPES)
    model = DeDoDeDescriptor(encoder=encoder, decoder=decoder).to(device)
    if weights is not None:
        model.load_state_dict(weights)
    return model
