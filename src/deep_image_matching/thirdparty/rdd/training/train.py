import argparse
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import tqdm
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from RDD.RDD import build
from RDD.RDD_helper import RDD_helper
from RDD.dataset.megadepth import megadepth_warper
from RDD.dataset.megadepth.megadepth import MegaDepthDataset
from RDD.utils import read_config
from benchmarks.mega_1500 import MegaDepthPoseMNNBenchmark
from training.losses.descriptor_loss import DescriptorLoss
from training.losses.detector_loss import DetectorLoss, compute_correspondence
from training.utils import check_accuracy

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_arguments():
    parser = argparse.ArgumentParser(description="RDD training script.")
    parser.add_argument('--megadepth_root_path', type=str, default='./data/megadepth',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--test_data_root', type=str, default='./data/megadepth_test_1500',
                        help='Path to the MegaDepth test dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the checkpoints.')
    parser.add_argument('--model_name', type=str, default='RDD',
                        help='Name of the model to save.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training. Default is 4.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate. Default is 0.0001.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=int, default=800,
                        help='Training resolution as width,height. Default is 800 for training descriptor.')
    parser.add_argument('--save_ckpt_every', type=int, default=500,
                        help='Save checkpoints every N steps. Default is 500.')
    parser.add_argument('--test_every_iter', type=int, default=2000,
                        help='Run validation every N steps. Default is 2000.')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to a checkpoint to resume from.')
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--enc_n_points', type=int, default=8)
    parser.add_argument('--num_feature_levels', type=int, default=5)
    parser.add_argument('--train_detector', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--config_path', type=str, default='./configs/default.yaml')
    parser.add_argument('--seed', type=int, default=0,
                        help='Base random seed to make training reproducible.')
    return parser.parse_args()


class Trainer:
    def __init__(self, rank, args):
        self.rank = rank
        self.args = args
        self.is_main_process = rank == 0
        self.is_distributed = args.distributed

        self.seed = args.seed
        self.epoch = 0
        self.training_res = args.training_res

        self._set_seed(self.seed)
        self.device, self.batch_size = self._init_device()

        config = self._prepare_config()
        self.model = self._build_model(config)

        self.opt = optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=args.lr,
            weight_decay=1e-4
        )
        self.scheduler = self._build_scheduler()

        if args.train_detector:
            self.detector_loss = DetectorLoss(temperature=0.1, scores_th=0.1)
        else:
            self.descriptor_loss = DescriptorLoss(inv_temp=20)

        self.benchmark = MegaDepthPoseMNNBenchmark(data_root=args.test_data_root)

        self.megadepth_root = Path(args.megadepth_root_path)
        self.train_base_path = self.megadepth_root / "megadepth_indices"
        train_npz_root = self.train_base_path / "scene_info_0.1_0.7"
        self.npz_paths = list(train_npz_root.glob("*.npz"))
        print('Loading MegaDepth dataset from', self.train_base_path)

        self.create_data_loader()

        self.ckpt_save_path = args.ckpt_save_path
        self.ckpt_save_path.mkdir(parents=True, exist_ok=True)
        log_dir = self.ckpt_save_path / 'logdir'
        log_dir.mkdir(parents=True, exist_ok=True)

        self.save_ckpt_every = args.save_ckpt_every
        self.model_name = args.model_name
        self.writer = None
        if self.is_main_process:
            run_name = f"{args.model_name}_{time.strftime('%Y_%m_%d-%H_%M_%S')}"
            self.writer = SummaryWriter(str(log_dir / run_name))

        self.saved_ckpts = []
        self.best = -1.0
        self.best_loss = 1e6
        self.fine_weight = 1.0
        self.dual_softmax_weight = 1.0
        self.heatmaps_weight = 1.0

    def create_data_loader(self):
        dataset = self._build_megadepth_dataset()

        if self.is_distributed:
            sampler = DistributedSampler(dataset, rank=self.rank, num_replicas=self.n_gpus, seed=self.seed)
        else:
            sampler = RandomSampler(dataset, generator=self._build_generator(self.seed))

        self.data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=self._seed_worker,
            generator=self._build_generator(self.seed)
        )

    def validate(self, total_steps):
        method = 'sparse' if self.args.train_detector else 'aliked'

        with torch.no_grad():
            model = self.model.module if self.is_distributed else self.model
            model.eval()
            model_helper = RDD_helper(model)
            test_out = self.benchmark.benchmark(
                model_helper,
                model_name='experiment',
                plot_every_iter=1,
                plot=False,
                method=method
            )
            if self.args.train_detector:
                model.set_softdetect(top_k=500, scores_th=0.2)

        auc5 = test_out['auc_5']
        auc10 = test_out['auc_10']
        auc20 = test_out['auc_20']

        if self.is_main_process and self.writer is not None:
            self.writer.add_scalar('Accuracy/auc5', auc5, total_steps)
            self.writer.add_scalar('Accuracy/auc10', auc10, total_steps)
            self.writer.add_scalar('Accuracy/auc20', auc20, total_steps)
            if auc5 > self.best:
                self.best = auc5
                self._save_checkpoint_state('best')

        self.model.train()

    def train(self):
        self.model.train()
        self.stride = 4 if self.args.num_feature_levels == 5 else 8
        total_steps = 0

        for epoch in range(self.args.epochs):
            if self.is_distributed and hasattr(self.data_loader.sampler, "set_epoch"):
                self.data_loader.sampler.set_epoch(epoch)

            pbar = tqdm.tqdm(
                total=len(self.data_loader),
                desc=f"Epoch {epoch + 1}/{self.args.epochs}"
            ) if self.is_main_process else None

            for _, batch in enumerate(self.data_loader):
                loss, acc_coarse, acc_kp, nb_coarse = self._inference(batch)

                if loss is None:
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()

                if (total_steps + 1) % self.save_ckpt_every == 0 and self.is_main_process:
                    print('saving iter ', total_steps + 1)
                    self._save_checkpoint_state(str(total_steps + 1))
                    self.saved_ckpts.append(total_steps + 1)
                    if len(self.saved_ckpts) > 5:
                        oldest_step = self.saved_ckpts.pop(0)
                        oldest_ckpt = self.ckpt_save_path / f'{self.model_name}_{oldest_step}.pth'
                        if oldest_ckpt.exists():
                            oldest_ckpt.unlink()

                if self.is_distributed:
                    torch.distributed.barrier()

                if (total_steps + 1) % self.args.test_every_iter == 0:
                    self.validate(total_steps)

                if pbar is not None:
                    if self.args.train_detector:
                        pbar.set_description(f'Loss: {loss.item():.4f}')
                    else:
                        pbar.set_description(
                            f'Loss: {loss.item():.4f} acc_coarse {acc_coarse:.3f} '
                            f'acc_kp: {acc_kp:.3f} #matches_c: {nb_coarse:d}'
                        )
                    pbar.update(1)

                if self.is_main_process and self.writer is not None:
                    self.writer.add_scalar('Loss/total', loss.item(), total_steps)
                    self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, total_steps)
                    self.writer.add_scalar('Count/matches_coarse', nb_coarse, total_steps)

                if not self.is_distributed:
                    self.scheduler.step()

                total_steps += 1

            self.validate(total_steps)
            if self.is_main_process:
                print('Epoch', epoch, 'done.')
                print('Creating new data loader with seed', self.seed)
            self.seed += 1
            self.set_seed(self.seed)
            self.scheduler.step()
            self.epoch += 1
            self.create_data_loader()

    def set_seed(self, seed):
        self._set_seed(seed)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _prepare_config(self):
        config = read_config(self.args.config_path)
        config['num_encoder_layers'] = self.args.num_encoder_layers
        config['enc_n_points'] = self.args.enc_n_points
        config['num_feature_levels'] = self.args.num_feature_levels
        config['train_detector'] = self.args.train_detector
        config['weights'] = self.args.weights
        config['device'] = self.device
        return config

    def _init_device(self):
        if self.is_distributed:
            print(f"Training in distributed mode with {self.args.n_gpus} GPUs")
            assert torch.cuda.is_available()
            device = self.rank
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=self.args.n_gpus,
                rank=device,
                init_method=f"file://{self.args.lock_file}",
                timeout=timedelta(seconds=2000)
            )
            torch.cuda.set_device(device)
            batch_size = int(self.args.batch_size / self.args.n_gpus)
            self.n_gpus = self.args.n_gpus
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch_size = self.args.batch_size
            self.n_gpus = 1
        print(f"Using device {device}")
        return device, batch_size

    def _build_model(self, config):
        model = build(config)

        if self.args.train_detector:
            model.set_softdetect(top_k=500, scores_th=0.2)

        if self.args.weights is not None:
            print('Loading weights from', self.args.weights)
            model.load_state_dict(torch.load(self.args.weights, map_location='cpu'))

        if self.is_distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.device], find_unused_parameters=True
            )
        else:
            model = model.to(self.device)

        return model

    def _build_scheduler(self):
        if self.is_distributed:
            return MultiStepLR(self.opt, milestones=[2, 4, 8, 16], gamma=self.args.gamma_steplr)
        return StepLR(self.opt, step_size=self.args.test_every_iter, gamma=self.args.gamma_steplr)

    def _build_megadepth_dataset(self):
        if self.args.train_detector:
            min_overlap, max_overlap, num_per_scene = 0.1, 0.8, 100
        else:
            min_overlap, max_overlap, num_per_scene = 0.01, 0.7, 200

        datasets = []
        root = str(self.megadepth_root)
        for path in self.npz_paths:
            for crop_or_scale in ('crop', 'scale'):
                datasets.append(
                    MegaDepthDataset(
                        root=root,
                        npz_path=str(path),
                        min_overlap_score=min_overlap,
                        max_overlap_score=max_overlap,
                        image_size=self.training_res,
                        num_per_scene=num_per_scene,
                        gray=False,
                        crop_or_scale=crop_or_scale
                    )
                )
        return ConcatDataset(datasets)

    def _save_checkpoint_state(self, suffix):
        if not self.is_main_process:
            return
        target_path = self.ckpt_save_path / f'{self.model_name}_{suffix}.pth'
        state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        torch.save(state_dict, target_path)

    def _build_generator(self, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def _inference(self, batch):
        if batch is not None:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            p1, p2 = batch['image0'], batch['image1']

            if not self.args.train_detector:
                positives_md_coarse = megadepth_warper.spvs_coarse(batch, self.stride)

        with torch.no_grad():
            if not self.args.train_detector:
                positives_c = positives_md_coarse

        is_corrupted = False
        if not self.args.train_detector:
            for positives in positives_c:
                if len(positives) < 30:
                    is_corrupted = True

        if is_corrupted:
            return None, None, None, None

        feats1, scores_map1, hmap1 = self.model(p1)
        feats2, scores_map2, hmap2 = self.model(p2)

        if self.args.train_detector:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            batch[key][sub_key] = sub_value.to(self.device)

            pred0 = {
                'descriptor_map': F.interpolate(feats1, size=scores_map1.shape[-2:], mode='bilinear', align_corners=True),
                'scores_map': scores_map1
            }
            pred1 = {
                'descriptor_map': F.interpolate(feats2, size=scores_map2.shape[-2:], mode='bilinear', align_corners=True),
                'scores_map': scores_map2
            }
            if self.is_distributed:
                correspondences, pred0_with_rand, pred1_with_rand = compute_correspondence(
                    self.model.module, pred0, pred1, batch, debug=True
                )
            else:
                correspondences, pred0_with_rand, pred1_with_rand = compute_correspondence(
                    self.model, pred0, pred1, batch, debug=False
                )

            loss = self.detector_loss(correspondences, pred0_with_rand, pred1_with_rand)
            acc_coarse, acc_kp, nb_coarse = 0, 0, 0
        else:
            loss_items = []
            acc_coarse_items = []
            acc_kp_items = []

            for idx, positives in enumerate(positives_c):
                if len(positives) > 10000:
                    positives = positives[torch.randperm(len(positives))[:10000]]

                pts1, pts2 = positives[:, :2], positives[:, 2:]

                h1 = hmap1[idx, :, :, :]
                h2 = hmap2[idx, :, :, :]

                m1 = feats1[idx, :, pts1[:, 1].long(), pts1[:, 0].long()].permute(1, 0)
                m2 = feats2[idx, :, pts2[:, 1].long(), pts2[:, 0].long()].permute(1, 0)
                loss_ds, loss_h, acc_kp = self.descriptor_loss(m1, m2, h1, h2, pts1, pts2)

                loss_items.append(loss_ds.unsqueeze(0) + loss_h.unsqueeze(0))

                acc_coarse = check_accuracy(m1, m2)
                acc_coarse_items.append(acc_coarse)
                acc_kp_items.append(acc_kp)

            nb_coarse = len(m1)
            loss = torch.cat(loss_items, -1).mean()
            acc_coarse = sum(acc_coarse_items) / len(acc_coarse_items)
            acc_kp = sum(acc_kp_items) / len(acc_kp_items)

        return loss, acc_coarse, acc_kp, nb_coarse


def main_worker(rank, args):
    trainer = Trainer(rank=rank, args=args)
    trainer.train()


def main():
    args = parse_arguments()

    if args.distributed:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)

    ckpt_path = Path(args.ckpt_save_path)
    if not ckpt_path.exists():
        os.makedirs(ckpt_path)
    args.ckpt_save_path = ckpt_path.resolve()

    if args.distributed:
        args.n_gpus = torch.cuda.device_count()
        args.lock_file = Path(args.ckpt_save_path) / "distributed_lock"
        if args.lock_file.exists():
            args.lock_file.unlink()

        torch.multiprocessing.spawn(main_worker, nprocs=args.n_gpus, args=(args,))
    else:
        main_worker(0, args)


if __name__ == '__main__':
    main()
