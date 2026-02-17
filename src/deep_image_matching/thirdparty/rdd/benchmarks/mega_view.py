import sys
sys.path.append(".")
import numpy as np
import torch
from PIL import Image
import tqdm
import cv2
import argparse
import matplotlib.pyplot as plt
import matplotlib
from RDD.RDD import build
from RDD.RDD_helper import RDD_helper
import os
from benchmarks.utils import pose_auc, angle_error_vec, angle_error_mat, symmetric_epipolar_distance, compute_symmetrical_epipolar_errors, compute_pose_error, compute_relative_pose, estimate_pose, dynamic_alpha

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig

def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)

def _make_evaluation_figure(img0, img1, kpts0, kpts1, epi_errs, e_t, e_R, alpha='dynamic', path=None):
    conf_thr = 1e-4
    
    img0 = np.array(img0)
    img1 = np.array(img1)
    
    kpts0 = kpts0
    kpts1 = kpts1
    
    epi_errs = epi_errs.cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)
    
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'e_t: {e_t:.2f} | e_R: {e_R:.2f}',
    ]
    
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text, path=path)
    return figure

class MegaDepthPoseMNNBenchmark:
    def __init__(self, data_root="./megadepth_test_1500", scene_names = None) -> None:
        if scene_names is None:
            self.scene_names = [
                "hard_indices.npz",
            ]
            # self.scene_names = ["0022_0.5_0.7.npz",]
        else:
            self.scene_names = scene_names
        self.scenes = [
            np.load(f"{data_root}/{scene}", allow_pickle=True)
            for scene in self.scene_names
        ]
        self.data_root = data_root

    def benchmark(self, model_helper, model_name = None, scale_intrinsics = False, calibrated = True, plot_every_iter=1, plot=False, method='sparse'):
        with torch.no_grad():
            data_root = self.data_root
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            thresholds = [5, 10, 20]
            for scene_ind in range(len(self.scenes)):
                scene_name = os.path.splitext(self.scene_names[scene_ind])[0]
                scene = self.scenes[scene_ind]
                indices = scene['indices']
                idx = 0
    
                for pair in tqdm.tqdm(indices):
                    
                    pairs = pair['pair_names']
                    K0 = pair['intrisinic'][0].copy().astype(np.float32)
                    T0 = pair['pose'][0].copy().astype(np.float32)
                    R0, t0 = T0[:3, :3], T0[:3, 3]
                    K1 = pair['intrisinic'][1].copy().astype(np.float32)    
                    T1 = pair['pose'][1].copy().astype(np.float32)
                    R1, t1 = T1[:3, :3], T1[:3, 3]
                    R, t = compute_relative_pose(R0, t0, R1, t1)
                    T0_to_1 = np.concatenate((R,t[:,None]), axis=-1)
                    im_A_path = f"{data_root}/images/{pairs[0]}"
                    im_B_path = f"{data_root}/images/{pairs[1]}"
                    
                    im_A = cv2.imread(im_A_path)
                    im_B = cv2.imread(im_B_path)

                    if method == 'dense':
                        kpts0, kpts1, conf = model_helper.match_dense(im_A, im_B, thr=0.01, resize=1600)
                    elif method == 'lightglue':
                        kpts0, kpts1, conf = model_helper.match_lg(im_A, im_B, thr=0.01,  resize=1600)
                    elif method == 'sparse':
                        kpts0, kpts1, conf = model_helper.match(im_A, im_B, thr=0.01,  resize=1600)
                    else:
                        raise ValueError(f"Invalid method {method}")
            
                    im_A = Image.open(im_A_path)
                    w0, h0 = im_A.size
                    im_B = Image.open(im_B_path)
                    w1, h1 = im_B.size
                    
                    if scale_intrinsics:
                        scale0 = 840 / max(w0, h0)
                        scale1 = 840 / max(w1, h1)
                        w0, h0 = scale0 * w0, scale0 * h0
                        w1, h1 = scale1 * w1, scale1 * h1
                        K0, K1 = K0.copy(), K1.copy()
                        K0[:2] = K0[:2] * scale0
                        K1[:2] = K1[:2] * scale1
                    
                    threshold = 0.5 
                    if calibrated:
                        norm_threshold = threshold / (np.mean(np.abs(K0[:2, :2])) + np.mean(np.abs(K1[:2, :2])))
                        ret = estimate_pose(
                            kpts0,
                            kpts1,
                            K0,
                            K1,
                            norm_threshold,
                            conf=0.99999,
                        )
                    if ret is not None:
                        R_est, t_est, mask = ret
                        T0_to_1_est = np.concatenate((R_est, t_est), axis=-1)  #
                        T0_to_1 = np.concatenate((R, t[:,None]), axis=-1)
                        e_t, e_R = compute_pose_error(T0_to_1_est, R, t)
                        
                        epi_errs = compute_symmetrical_epipolar_errors(T0_to_1, kpts0, kpts1, K0, K1)
                        if scene_ind % plot_every_iter == 0 and plot:

                            if not os.path.exists(f'outputs/mega_view/{model_name}_{method}'):
                                os.mkdir(f'outputs/mega_view/{model_name}_{method}')
                            name = f'outputs/mega_view/{model_name}_{method}/{scene_name}_{idx}.png'
                            _make_evaluation_figure(im_A, im_B, kpts0, kpts1, epi_errs, e_t, e_R, path=name)
                        e_pose = max(e_t, e_R)
                    else:
                        e_t, e_R = np.inf, np.inf  # or any large sentinel value to indicate failure
                        e_pose = max(e_t, e_R)
                    tot_e_t.append(e_t)
                    tot_e_R.append(e_R)
                    tot_e_pose.append(e_pose)
                    idx += 1
                    
            tot_e_pose = np.array(tot_e_pose)
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
            print(f"{model_name} auc: {auc}")
            return {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
            
    
            
def parse_arguments():
    parser = argparse.ArgumentParser(description="Testing script.")
    
    parser.add_argument("--data_root", type=str, default="./data/megadepth_view", help="Path to the MegaDepth dataset.")

    parser.add_argument("--weights", type=str, default="./weights/RDD-v2.pth", help="Path to the model checkpoint.")

    parser.add_argument("--plot", action="store_true", help="Whether to plot the results.")

    parser.add_argument("--method", type=str, default="sparse", help="Method for matching.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()    
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    
    if not os.path.exists(f'outputs/mega_view'):
        os.mkdir(f'outputs/mega_view')
    model = build(weights=args.weights)
    benchmark = MegaDepthPoseMNNBenchmark(data_root=args.data_root)
    model.eval()
    model_helper = RDD_helper(model)
    with torch.no_grad():
        method = args.method
        out = benchmark.benchmark(model_helper, model_name='RDD', plot_every_iter=1, plot=args.plot, method=method)
        with open(f'outputs/mega_view/RDD_{method}.txt', 'w') as f:
            f.write(str(out))


