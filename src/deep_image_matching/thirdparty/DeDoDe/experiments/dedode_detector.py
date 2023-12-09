import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset
import torch.nn as nn

from DeDoDe.train import train_k_steps
from DeDoDe.datasets.megadepth import MegadepthBuilder
from DeDoDe.detectors.keypoint_loss import KeyPointLoss
from DeDoDe.checkpoint import CheckPoint
from DeDoDe.detectors.dedode_detector import DeDoDeDetector
from DeDoDe.encoder import VGG19
from DeDoDe.decoder import Decoder, ConvRefiner
from DeDoDe.benchmarks import NumInliersBenchmark

def train():
    NUM_PROTOTYPES = 1
    residual = True
    hidden_blocks = 8
    amp_dtype = torch.float16
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,
            ),
            "4": ConvRefiner(
                256+256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,

            ),
            "2": ConvRefiner(
                128+128,
                128,
                64 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,

            ),
            "1": ConvRefiner(
                64 + 64,
                64,
                1 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,
            ),
        }
    )
    import os
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    encoder = VGG19(pretrained = True, amp = amp, amp_dtype = amp_dtype)
    decoder = Decoder(conv_refiner, num_prototypes=NUM_PROTOTYPES)
    model = DeDoDeDetector(encoder = encoder, decoder = decoder).cuda()
    params = [
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.decoder.parameters(), "lr": 2e-4},
        ]
    optim = AdamW(params, weight_decay = 1e-5)
    n0, N, k = 0, 100_000, 1000
    lr_scheduler = CosineAnnealingLR(optim, T_max = N)
    checkpointer = CheckPoint("workspace/", name = experiment_name)
    
    model, optim, lr_scheduler, n0 = checkpointer.load(model, optim, lr_scheduler, n0)
    
    loss = KeyPointLoss(smoothing_size = 51)
    
    H, W = 512, 512
    mega = MegadepthBuilder(data_root="data/megadepth", loftr_ignore=True, imc21_ignore = True)
    use_horizontal_flip_aug = False
    megadepth_train1 = mega.build_scenes(
        split="train_loftr", min_overlap=0.01, ht=H, wt=W, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug,
    )
    megadepth_train2 = mega.build_scenes(
        split="train_loftr", min_overlap=0.35, ht=H, wt=W, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug,
    )

    megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
    mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)
    
    megadepth_test = mega.build_scenes(
        split="test_loftr", min_overlap=0.01, ht=H, wt=W, 
        shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug,
    )
    mega_test = NumInliersBenchmark(ConcatDataset(megadepth_test))
    grad_scaler = torch.cuda.amp.GradScaler()
    
    for n in range(n0, N, k):
        mega_sampler = torch.utils.data.WeightedRandomSampler(
            mega_ws, num_samples = 8 * k, replacement=False
        )
        mega_dataloader = iter(
            torch.utils.data.DataLoader(
                megadepth_train,
                batch_size = 8,
                sampler = mega_sampler,
                num_workers = 8,
            )
        )
        train_k_steps(
            n, k, mega_dataloader, model, loss, optim, lr_scheduler, grad_scaler = grad_scaler,
        )
        checkpointer.save(model, optim, lr_scheduler, n)
        mega_test.benchmark(model)


if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    train()