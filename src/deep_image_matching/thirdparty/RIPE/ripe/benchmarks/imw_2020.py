import os
from pathlib import Path

import cv2
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import poselib
import torch
from tqdm import tqdm

from ripe import utils
from ripe.data.data_transforms import Compose, Normalize, Resize
from ripe.data.datasets.disk_imw import DISK_IMW
from ripe.utils.pose_error import AUCMetric, relative_pose_error
from ripe.utils.utils import (
    cv2_matches_from_kornia,
    cv_resize_and_pad_to_shape,
    to_cv_kpts,
)

log = utils.get_pylogger(__name__)


class IMW_2020_Benchmark:
    def __init__(
        self,
        use_predefined_subset: bool = True,
        conf_inference=None,
        edge_input_divisible_by=None,
    ):
        data_dir = os.getenv("DATA_DIR")
        if data_dir is None:
            raise ValueError("Environment variable DATA_DIR is not set.")
        root_path = Path(data_dir) / "disk-data"

        self.data = DISK_IMW(
            str(
                root_path
            ),  # Resize only to ensure that the input size is divisible the value of edge_input_divisible_by
            transforms=Compose(
                [
                    Resize(None, edge_input_divisible_by),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
        self.ids_subset = None
        self.results = []
        self.conf_inference = conf_inference

        # fmt: off
        if use_predefined_subset:
            self.ids_subset = [4921, 3561, 3143, 6040, 802, 6828, 5338, 9275, 10764, 10085, 5124, 11355, 7, 10027, 2161, 4433, 6887, 3311, 10766,
                               11451, 11433, 8539, 2581, 10300, 10562, 1723, 8803, 6275, 10140, 11487, 6238, 638, 8092, 9979, 201, 10394, 3414,
                               9002, 7456, 2431, 632, 6589, 9265, 9889, 3139, 7890, 10619, 4899, 675, 176, 4309, 4814, 3833, 3519, 148, 4560, 10705,
                               3744, 1441, 4049, 1791, 5106, 575, 1540, 1105, 6791, 1383, 9344, 501, 2504, 4335, 8992, 10970, 10786, 10405, 9317,
                               5279, 1396, 5044, 9408, 11125, 10417, 7627, 7480, 1358, 7738, 5461, 10178, 9226, 8106, 2766, 6216, 4032, 7298, 259,
                               3021, 2645, 8756, 7513, 3163, 2510, 6701, 6684, 3159, 9689, 7425, 6066, 1904, 6382, 3052, 777, 6277, 7409, 5997, 2987,
                               11316, 2894, 4528, 1927, 10366, 8605, 2726, 1886, 2416, 2164, 3352, 2997, 6636, 6765, 5609, 3679, 76, 10956, 3612, 6699,
                               1741, 8811, 3755, 1285, 9520, 2476, 3977, 370, 9823, 1834, 7551, 6227, 7303, 6399, 4758, 10713, 5050, 380, 11056, 7620,
                               4826, 6090, 9011, 7523, 7355, 8021, 9801, 1801, 6522, 7138, 10017, 8732, 6402, 3116, 4031, 6088, 3975, 9841, 9082, 9412,
                               5406, 217, 2385, 8791, 8361, 494, 4319, 5275, 3274, 335, 6731, 207, 10095, 3068, 5996, 3951, 2808, 5877, 6134, 7772, 10042,
                               8574, 5501, 10885, 7871]
            # self.ids_subset = self.ids_subset[:10]
        # fmt: on

    def evaluate_sample(self, model, sample, dev):
        img_1 = sample["src_image"].unsqueeze(0).to(dev)
        img_2 = sample["trg_image"].unsqueeze(0).to(dev)

        scale_h_1, scale_w_1 = (
            sample["orig_size_src"][0] / img_1.shape[2],
            sample["orig_size_src"][1] / img_1.shape[3],
        )
        scale_h_2, scale_w_2 = (
            sample["orig_size_trg"][0] / img_2.shape[2],
            sample["orig_size_trg"][1] / img_2.shape[3],
        )

        M = None
        info = {}
        kpts_1, desc_1, score_1 = None, None, None
        kpts_2, desc_2, score_2 = None, None, None
        match_dists, match_idxs = None, None

        try:
            kpts_1, desc_1, score_1 = model.detectAndCompute(img_1, **self.conf_inference)
            kpts_2, desc_2, score_2 = model.detectAndCompute(img_2, **self.conf_inference)

            if kpts_1.dim() == 3:
                assert kpts_1.shape[0] == 1 and kpts_2.shape[0] == 1, "Batch size must be 1"

                kpts_1, desc_1, score_1 = (
                    kpts_1.squeeze(0),
                    desc_1[0].squeeze(0),
                    score_1[0].squeeze(0),
                )
                kpts_2, desc_2, score_2 = (
                    kpts_2.squeeze(0),
                    desc_2[0].squeeze(0),
                    score_2[0].squeeze(0),
                )

            scale_1 = torch.tensor([scale_w_1, scale_h_1], dtype=torch.float).to(dev)
            scale_2 = torch.tensor([scale_w_2, scale_h_2], dtype=torch.float).to(dev)

            kpts_1 = kpts_1 * scale_1
            kpts_2 = kpts_2 * scale_2

            matcher = KF.DescriptorMatcher("mnn")  # threshold is not used with mnn
            match_dists, match_idxs = matcher(desc_1, desc_2)

            matched_pts_1 = kpts_1[match_idxs[:, 0]]
            matched_pts_2 = kpts_2[match_idxs[:, 1]]

            camera_1 = sample["src_camera"]
            camera_2 = sample["trg_camera"]

            M, info = poselib.estimate_relative_pose(
                matched_pts_1.cpu().numpy(),
                matched_pts_2.cpu().numpy(),
                camera_1.to_cameradict(),
                camera_2.to_cameradict(),
                {
                    "max_epipolar_error": 0.5,
                },
                {},
            )
        except RuntimeError as e:
            if "No keypoints detected" in str(e):
                pass
            else:
                raise e

        success = M is not None
        if success:
            M = {
                "R": torch.tensor(M.R, dtype=torch.float),
                "t": torch.tensor(M.t, dtype=torch.float),
            }
            inl = info["inliers"]
        else:
            M = {
                "R": torch.eye(3, dtype=torch.float),
                "t": torch.zeros((3), dtype=torch.float),
            }
            inl = np.zeros((0,)).astype(bool)

        t_err, r_err = relative_pose_error(sample["s2t_R"].cpu(), sample["s2t_T"].cpu(), M["R"], M["t"])

        rel_pose_error = max(t_err.item(), r_err.item()) if success else np.inf
        ransac_inl = np.sum(inl)
        ransac_inl_ratio = np.mean(inl)

        if success:
            assert match_dists is not None and match_idxs is not None, "Matches must be computed"
            cv_keypoints_src = to_cv_kpts(kpts_1, score_1)
            cv_keypoints_trg = to_cv_kpts(kpts_2, score_2)
            cv_matches = cv2_matches_from_kornia(match_dists, match_idxs)
            cv_mask = [int(m) for m in inl]
        else:
            cv_keypoints_src, cv_keypoints_trg = [], []
            cv_matches, cv_mask = [], []

        estimation = {
            "success": success,
            "M_0to1": M,
            "inliers": torch.tensor(inl).to(img_1),
            "rel_pose_error": rel_pose_error,
            "ransac_inl": ransac_inl,
            "ransac_inl_ratio": ransac_inl_ratio,
            "path_src_image": sample["src_path"],
            "path_trg_image": sample["trg_path"],
            "cv_keypoints_src": cv_keypoints_src,
            "cv_keypoints_trg": cv_keypoints_trg,
            "cv_matches": cv_matches,
            "cv_mask": cv_mask,
        }

        return estimation

    def evaluate(self, model, dev, progress_bar=False):
        model.eval()

        # reset results
        self.results = []

        for idx in tqdm(
            self.ids_subset if self.ids_subset is not None else range(len(self.data)),
            disable=not progress_bar,
        ):
            sample = self.data[idx]
            self.results.append(self.evaluate_sample(model, sample, dev))

    def get_auc(self, threshold=5, downsampled=False):
        if len(self.results) == 0:
            raise ValueError("No results to log. Run evaluate first.")

        summary_results = self.calc_auc(downsampled=downsampled)

        return summary_results[f"rel_pose_error@{threshold}°{'__original' if not downsampled else '__downsampled'}"]

    def plot_results(self, num_samples=10, logger=None, step=None, downsampled=False):
        if len(self.results) == 0:
            raise ValueError("No results to plot. Run evaluate first.")

        plot_data = []

        for result in self.results[:num_samples]:
            img1 = cv2.imread(result["path_src_image"])
            img2 = cv2.imread(result["path_trg_image"])

            # from BGR to RGB
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            plt_matches = cv2.drawMatches(
                img1,
                result["cv_keypoints_src"],
                img2,
                result["cv_keypoints_trg"],
                result["cv_matches"],
                None,
                matchColor=None,
                matchesMask=result["cv_mask"],
                flags=cv2.DrawMatchesFlags_DEFAULT,
            )
            file_name = (
                Path(result["path_src_image"]).parent.parent.name
                + "_"
                + Path(result["path_src_image"]).stem
                + Path(result["path_trg_image"]).stem
                + ("_downsampled" if downsampled else "")
                + ".png"
            )
            # print rel_pose_error on image
            plt_matches = cv2.putText(
                plt_matches,
                f"rel_pose_error: {result['rel_pose_error']:.2f} num_inliers: {result['ransac_inl']} inl_ratio: {result['ransac_inl_ratio']:.2f} num_matches: {len(result['cv_matches'])} num_keypoints: {len(result['cv_keypoints_src'])}/{len(result['cv_keypoints_trg'])}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
                cv2.LINE_8,
            )

            plot_data.append({"file_name": file_name, "image": plt_matches})

        if logger is None:
            log.info("No logger provided. Using plt to plot results.")
            for image in plot_data:
                plt.imsave(
                    image["file_name"],
                    cv_resize_and_pad_to_shape(image["image"], (1024, 2048)),
                )
                plt.close()
        else:
            import wandb

            log.info(f"Logging images to wandb with step={step}")
            if not downsampled:
                logger.log(
                    {
                        "examples": [
                            wandb.Image(cv_resize_and_pad_to_shape(image["image"], (1024, 2048))) for image in plot_data
                        ]
                    },
                    step=step,
                )
            else:
                logger.log(
                    {
                        "examples_downsampled": [
                            wandb.Image(cv_resize_and_pad_to_shape(image["image"], (1024, 2048))) for image in plot_data
                        ]
                    },
                    step=step,
                )

    def log_results(self, logger=None, step=None, downsampled=False):
        if len(self.results) == 0:
            raise ValueError("No results to log. Run evaluate first.")

        summary_results = self.calc_auc(downsampled=downsampled)

        if logger is not None:
            logger.log(summary_results, step=step)
        else:
            log.warning("No logger provided. Printing results instead.")
            print(self.calc_auc())

    def print_results(self):
        if len(self.results) == 0:
            raise ValueError("No results to print. Run evaluate first.")

        print(self.calc_auc())

    def calc_auc(self, auc_thresholds=None, downsampled=False):
        if auc_thresholds is None:
            auc_thresholds = [5, 10, 20]
        if not isinstance(auc_thresholds, list):
            auc_thresholds = [auc_thresholds]

        if len(self.results) == 0:
            raise ValueError("No results to calculate auc. Run evaluate first.")

        rel_pose_errors = [r["rel_pose_error"] for r in self.results]

        pose_aucs = AUCMetric(auc_thresholds, rel_pose_errors).compute()
        assert isinstance(pose_aucs, list) and len(pose_aucs) == len(auc_thresholds)

        ext = "_downsampled" if downsampled else "_original"

        summary = {}
        for i, ath in enumerate(auc_thresholds):
            summary[f"rel_pose_error@{ath}°_{ext}"] = pose_aucs[i]

        return summary
