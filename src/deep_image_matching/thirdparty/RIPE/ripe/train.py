import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

SEED = 32000

import collections
import os

import hydra
from hydra.utils import instantiate
from lightning.fabric import Fabric

print(SEED)
import random

os.environ["PYTHONHASHSEED"] = str(SEED)

import numpy as np
import torch
import tqdm
import wandb
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from ripe import utils
from ripe.benchmarks.imw_2020 import IMW_2020_Benchmark
from ripe.utils.utils import get_rewards
from ripe.utils.wandb_utils import get_flattened_wandb_cfg

log = utils.get_pylogger(__name__)
from pathlib import Path

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def unpack_batch(batch):
    src_image = batch["src_image"]
    trg_image = batch["trg_image"]
    trg_mask = batch["trg_mask"]
    src_mask = batch["src_mask"]
    label = batch["label"]
    H = batch["homography"]

    return src_image, trg_image, src_mask, trg_mask, H, label


@hydra.main(config_path="../conf/", config_name="config", version_base=None)
def train(cfg):
    """Main training function for the RIPE model."""
    #  Prepare model, data and hyperparms

    strategy = "ddp" if cfg.num_gpus > 1 else "auto"
    fabric = Fabric(
        accelerator="cuda",
        devices=cfg.num_gpus,
        precision=cfg.precision,
        strategy=strategy,
    )
    fabric.launch()

    output_dir = Path(cfg.output_dir)
    experiment_name = output_dir.parent.parent.parent.name
    run_id = output_dir.parent.parent.name
    timestamp = output_dir.parent.name + "_" + output_dir.name

    experiment_name = run_id + " " + timestamp + " " + experiment_name

    # setup logger
    wandb_logger = wandb.init(
        project=cfg.project_name,
        name=experiment_name,
        config=get_flattened_wandb_cfg(cfg),
        dir=cfg.output_dir,
        mode=cfg.wandb_mode,
    )

    min_nums_matches = {"homography": 4, "fundamental": 8, "fundamental_7pt": 7}
    min_num_matches = min_nums_matches[cfg.transformation_model]
    print(f"Minimum number of matches for {cfg.transformation_model} is {min_num_matches}")

    batch_size = cfg.batch_size
    steps = cfg.num_steps
    lr = cfg.lr

    num_grad_accs = (
        cfg.num_grad_accs
    )  # this performs grad accumulation to simulate larger batch size, set to 1 to disable;

    # instantiate dataset
    ds = instantiate(cfg.data)

    # prepare dataloader
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
        num_workers=cfg.num_workers,
    )
    dl = fabric.setup_dataloaders(dl)
    i_dl = iter(dl)

    # create matcher
    matcher = instantiate(cfg.matcher)

    if cfg.desc_loss_weight != 0.0:
        descriptor_loss = instantiate(cfg.descriptor_loss)
    else:
        log.warning(
            "Descriptor loss weight is 0.0, descriptor loss will not be used. 1x1 conv for descriptors will be deactivated!"
        )
        descriptor_loss = None

    upsampler = instantiate(cfg.upsampler) if "upsampler" in cfg else None

    # create network
    net = instantiate(cfg.network)(
        net=instantiate(cfg.backbones),
        upsampler=upsampler,
        descriptor_dim=cfg.descriptor_dim if descriptor_loss is not None else None,
        device=fabric.device,
    ).train()

    # get num parameters
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.info(f"Number of parameters: {num_params}")

    fp_penalty = cfg.fp_penalty  # small penalty for not finding a match
    kp_penalty = cfg.kp_penalty  # small penalty for low logprob keypoints

    opt_pi = AdamW(filter(lambda x: x.requires_grad, net.parameters()), lr=lr, weight_decay=1e-5)
    net, opt_pi = fabric.setup(net, opt_pi)

    if cfg.lr_scheduler:
        scheduler = instantiate(cfg.lr_scheduler)(optimizer=opt_pi, steps_init=0)
    else:
        scheduler = None

    val_benchmark = IMW_2020_Benchmark(
        use_predefined_subset=True,
        conf_inference=cfg.conf_inference,
        edge_input_divisible_by=None,
    )

    # mean average of skipped batches
    # this is used to monitor how many batches were skipped due to not enough keypoints
    # this is useful to detect if the model is not learning anything -> should be zero
    ma_skipped_batches = collections.deque(maxlen=100)

    opt_pi.zero_grad()

    # initialize scheduler
    alpha_scheduler = instantiate(cfg.alpha_scheduler)
    beta_scheduler = instantiate(cfg.beta_scheduler)
    inl_th_scheduler = instantiate(cfg.inl_th)

    # ======  Training Loop  ======
    # check if the model is in training mode
    net.train()

    with tqdm.tqdm(total=steps) as pbar:
        for i_step in range(steps):
            alpha = alpha_scheduler(i_step)
            beta = beta_scheduler(i_step)
            inl_th = inl_th_scheduler(i_step)

            if scheduler:
                scheduler.step()

            # Initialize vars for current step
            # We need to handle batching because the description can have arbitrary number of keypoints
            sum_reward_batch = 0
            sum_num_keypoints_1 = 0
            sum_num_keypoints_2 = 0
            loss = None
            loss_policy_stack = None
            loss_desc_stack = None
            loss_kp_stack = None

            try:
                batch = next(i_dl)
            except StopIteration:
                i_dl = iter(dl)
                batch = next(i_dl)

            p1, p2, mask_padding_1, mask_padding_2, Hs, label = unpack_batch(batch)

            (
                kpts1,
                logprobs1,
                selected_mask1,
                mask_padding_grid_1,
                logits_selected_1,
                out1,
            ) = net(p1, mask_padding_1, training=True)
            (
                kpts2,
                logprobs2,
                selected_mask2,
                mask_padding_grid_2,
                logits_selected_2,
                out2,
            ) = net(p2, mask_padding_2, training=True)

            # upsample coarse descriptors for all keypoints from the intermediate feature maps from the encoder
            desc_1 = net.get_descs(out1["coarse_descs"], p1, kpts1, p1.shape[2], p1.shape[3])
            desc_2 = net.get_descs(out2["coarse_descs"], p2, kpts2, p2.shape[2], p2.shape[3])

            if cfg.padding_filter_mode == "ignore":  # remove keypoints that are in padding
                batch_mask_selection_for_matching_1 = selected_mask1 & mask_padding_grid_1
                batch_mask_selection_for_matching_2 = selected_mask2 & mask_padding_grid_2
            elif cfg.padding_filter_mode == "punish":
                batch_mask_selection_for_matching_1 = selected_mask1  # keep all keypoints
                batch_mask_selection_for_matching_2 = selected_mask2  # punish the keypoints in the padding area
            else:
                raise ValueError(f"Unknown padding filter mode: {cfg.padding_filter_mode}")

            (
                batch_rel_idx_matches,
                batch_abs_idx_matches,
                batch_ransac_inliers,
                batch_Fm,
            ) = matcher(
                kpts1,
                kpts2,
                desc_1,
                desc_2,
                batch_mask_selection_for_matching_1,
                batch_mask_selection_for_matching_2,
                inl_th,
                label if cfg.no_filtering_negatives else None,
            )

            for b in range(batch_size):
                # ignore if less than 16 keypoints have been detected
                if batch_rel_idx_matches[b] is None:
                    ma_skipped_batches.append(1)
                    continue
                else:
                    ma_skipped_batches.append(0)

                mask_selection_for_matching_1 = batch_mask_selection_for_matching_1[b]
                mask_selection_for_matching_2 = batch_mask_selection_for_matching_2[b]

                rel_idx_matches = batch_rel_idx_matches[b]
                abs_idx_matches = batch_abs_idx_matches[b]
                ransac_inliers = batch_ransac_inliers[b]

                if cfg.selected_only:
                    # every SELECTED keypoint with every other SELECTED keypoint
                    dense_logprobs = logprobs1[b][mask_selection_for_matching_1].view(-1, 1) + logprobs2[b][
                        mask_selection_for_matching_2
                    ].view(1, -1)
                else:
                    if cfg.padding_filter_mode == "ignore":
                        # every keypoint with every other keypoint, but WITHOUT keypoint in the padding area
                        dense_logprobs = logprobs1[b][mask_padding_grid_1[b]].view(-1, 1) + logprobs2[b][
                            mask_padding_grid_2[b]
                        ].view(1, -1)
                    elif cfg.padding_filter_mode == "punish":
                        # every keypoint with every other keypoint, also WITH keypoints in the padding areas -> will be punished by the reward
                        dense_logprobs = logprobs1[b].view(-1, 1) + logprobs2[b].view(1, -1)
                    else:
                        raise ValueError(f"Unknown padding filter mode: {cfg.padding_filter_mode}")

                reward = None

                if cfg.reward_type == "inlier":
                    reward = (
                        0.5 if cfg.no_filtering_negatives and not label[b] else 1.0
                    )  # reward is 1.0 if the pair is positive, 0.5 if negative and no filtering is applied
                elif cfg.reward_type == "inlier_ratio":
                    ratio_inlier = ransac_inliers.sum() / len(abs_idx_matches)
                    reward = ratio_inlier  # reward is the ratio of inliers -> higher if more matches are inliers
                elif cfg.reward_type == "inlier+inlier_ratio":
                    ratio_inlier = ransac_inliers.sum() / len(abs_idx_matches)
                    reward = (
                        (1.0 - beta) * 1.0 + beta * ratio_inlier
                    )  # reward is a combination of the ratio of inliers and the number of inliers -> gradually changes
                else:
                    raise ValueError(f"Unknown reward type: {cfg.reward_type}")

                dense_rewards = get_rewards(
                    reward,
                    kpts1[b],
                    kpts2[b],
                    mask_selection_for_matching_1,
                    mask_selection_for_matching_2,
                    mask_padding_grid_1[b],
                    mask_padding_grid_2[b],
                    rel_idx_matches,
                    abs_idx_matches,
                    ransac_inliers,
                    label[b],
                    fp_penalty * alpha,
                    use_whitening=cfg.use_whitening,
                    selected_only=cfg.selected_only,
                    filter_mode=cfg.padding_filter_mode,
                )

                if descriptor_loss is not None:
                    hard_loss = descriptor_loss(
                        desc1=desc_1[b],
                        desc2=desc_2[b],
                        matches=abs_idx_matches,
                        inliers=ransac_inliers,
                        label=label[b],
                        logits_1=None,
                        logits_2=None,
                    )
                    loss_desc_stack = (
                        hard_loss if loss_desc_stack is None else torch.hstack((loss_desc_stack, hard_loss))
                    )

                sum_reward_batch += dense_rewards.sum()

                current_loss_policy = (dense_rewards * dense_logprobs).view(-1)

                loss_policy_stack = (
                    current_loss_policy
                    if loss_policy_stack is None
                    else torch.hstack((loss_policy_stack, current_loss_policy))
                )

                if kp_penalty != 0.0:
                    # keypoints with low logprob are penalized
                    # as they get large negative logprob values multiplying them with the penalty will make the loss larger
                    loss_kp = (
                        logprobs1[b][mask_selection_for_matching_1]
                        * torch.full_like(
                            logprobs1[b][mask_selection_for_matching_1],
                            kp_penalty * alpha,
                        )
                    ).mean() + (
                        logprobs2[b][mask_selection_for_matching_2]
                        * torch.full_like(
                            logprobs2[b][mask_selection_for_matching_2],
                            kp_penalty * alpha,
                        )
                    ).mean()
                    loss_kp_stack = loss_kp if loss_kp_stack is None else torch.hstack((loss_kp_stack, loss_kp))

                sum_num_keypoints_1 += mask_selection_for_matching_1.sum()
                sum_num_keypoints_2 += mask_selection_for_matching_2.sum()

            loss = loss_policy_stack.mean()
            if loss_kp_stack is not None:
                loss += loss_kp_stack.mean()

            loss = -loss

            if descriptor_loss is not None:
                loss += cfg.desc_loss_weight * loss_desc_stack.mean()

            pbar.set_description(
                f"LP: {loss.item():.4f} - Det: ({sum_num_keypoints_1 / batch_size:.4f}, {sum_num_keypoints_2 / batch_size:.4f}), #mRwd: {sum_reward_batch / batch_size:.1f}"
            )
            pbar.update()

            # backward pass
            loss /= num_grad_accs
            fabric.backward(loss)

            if i_step % num_grad_accs == 0:
                opt_pi.step()
                opt_pi.zero_grad()

            if i_step % cfg.log_interval == 0:
                wandb_logger.log(
                    {
                        # "loss": loss.item() if not use_amp else scaled_loss.item(),
                        "loss": loss.item(),
                        "loss_policy": -loss_policy_stack.mean().item(),
                        "loss_kp": loss_kp_stack.mean().item() if loss_kp_stack is not None else 0.0,
                        "loss_hard": (loss_desc_stack.mean().item() if loss_desc_stack is not None else 0.0),
                        "mean_num_det_kpts1": sum_num_keypoints_1 / batch_size,
                        "mean_num_det_kpts2": sum_num_keypoints_2 / batch_size,
                        "mean_reward": sum_reward_batch / batch_size,
                        "lr": opt_pi.param_groups[0]["lr"],
                        "ma_skipped_batches": sum(ma_skipped_batches) / len(ma_skipped_batches),
                        "inl_th": inl_th,
                    },
                    step=i_step,
                )

            if i_step % cfg.val_interval == 0:
                val_benchmark.evaluate(net, fabric.device, progress_bar=False)
                val_benchmark.log_results(logger=wandb_logger, step=i_step)

                # ensure that the model is in training mode again
                net.train()

    # save the model
    torch.save(
        net.state_dict(),
        output_dir / ("model" + "_" + str(i_step + 1) + "_final" + ".pth"),
    )


if __name__ == "__main__":
    train()
