import os
import numpy as np
import csv
import _pickle as cPickle
import bz2
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

def magadepth_intrinsics_statistics(bench_path):
    ppath = os.path.join(bench_path, 'gt_data', 'megadepth')
    npz_list = [i for i in os.listdir(ppath) if (os.path.splitext(i)[1] == '.npz')]

    data = {}
    for name in npz_list:
        scene_info = np.load(os.path.join(ppath, name), allow_pickle=True)
    
        # Collect pairs
        for pair_info in scene_info['pair_infos']:
            (id1, id2), _, _ = pair_info                
            K1 = scene_info['intrinsics'][id1].astype(np.float32)
            K2 = scene_info['intrinsics'][id2].astype(np.float32)
    
            idx1 = scene_info['image_paths'][id1].replace('Undistorted_SfM/', '')
            idx2 = scene_info['image_paths'][id2].replace('Undistorted_SfM/', '')
            data[idx1] = K1
            data[idx2] = K2

    return data

def scannet_intrinsics_statistics(bench_path):
    ppath = os.path.join(bench_path, 'gt_data', 'scannet')
    intrinsic_path = 'intrinsics.npz'
    npz_path = 'test.npz'

    data = np.load(os.path.join(ppath, npz_path))
    data_names = data['name']
    intrinsics = dict(np.load(os.path.join(ppath, intrinsic_path)))

    data = {}
    for idx in range(data_names.shape[0]):
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_names[idx]
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        K1 = intrinsics[scene_name]
        K2 = intrinsics[scene_name]

        idx1 = os.path.join(scene_name, 'color', f'{stem_name_0}.jpg')
        idx2 = os.path.join(scene_name, 'color', f'{stem_name_1}.jpg')
        data[idx1] = K1
        data[idx2] = K2

    return data


def phototourism_intrinsics_statistics(bench_path):
    out_dir = os.path.join(bench_path, 'imc_phototourism')
    scenes = [scene for scene in os.listdir(os.path.join(out_dir, 'train')) if os.path.isdir(os.path.join(out_dir, 'train', scene))]
    sampled_idx = decompress_pickle(os.path.join(bench_path, 'downloads', 'imc_sampled_idx.pbz2'))

    data = {}
    for sn in range(len(scenes)):
        scene = scenes[sn]
        work_path = os.path.join(out_dir, 'train', scene)
        pose_file  = os.path.join(work_path, 'calibration.csv')
        covis_file  = os.path.join(work_path, 'pair_covisibility.csv')

        if (not os.path.isfile(pose_file)) or (not os.path.isfile(covis_file)):
            continue

        im1_ = []
        im2_ = []
        with open(covis_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pp = row['pair'].split('-')
                im1_.append(os.path.join(scene, pp[0]))
                im2_.append(os.path.join(scene, pp[1]))
                # covis_val.append(float(row['covisibility']))

        full_idx = sampled_idx[scene]
        im1_ = [im1_[i] for i in full_idx]
        im2_ = [im2_[i] for i in full_idx]

        Kv = {}
        calib_file = os.path.join(work_path, 'calibration.csv')
        with open(calib_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cam = os.path.join(scene, row['image_id'])
                Kv[cam] = np.asarray([float(i) for i in row['camera_intrinsics'].split(' ')]).reshape((3, 3))

        for i in range(len(im1_)):
            K1 = Kv[im1_[i]]
            K2 = Kv[im2_[i]]
            
            idx1 = im1_[i]
            idx2 = im2_[i]
            data[idx1] = K1
            data[idx2] = K2

    return data

def plot_focal_length_vs_parameter(bench_path, data, dataset_name):
    ppath = os.path.join(bench_path, 'res', 'latex')
    os.makedirs(ppath, exist_ok=True)
    fig_dpi = 300

    focal_lengths = []
    q_values_min = []
    q_values_max = []
    q_values_avg = []
    q_values_sqrt = []

    for key, K in data.items():
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        q_min = min(cx, cy)
        q_max = max(cx, cy)
        q_avg = (cx + cy) / 2
        q_sqrt = np.sqrt(cx * cy)

        focal_length = max(fx, fy)

        focal_lengths.append(focal_length)
        q_values_min.append(q_min)
        q_values_max.append(q_max)
        q_values_avg.append(q_avg)
        q_values_sqrt.append(q_sqrt)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0, 0].scatter(q_values_min, focal_lengths, alpha=0.5, marker='.', s=3, color='r')
    axs[0, 0].set_title(dataset_name + ': $f$ vs $\min(c_x, c_y)$')
    axs[0, 0].set_xlabel('$\min(c_x, c_y)$')
    axs[0, 0].set_ylabel('$f$')

    axs[0, 1].scatter(q_values_max, focal_lengths, alpha=0.5, marker='.', s=3, color='g')
    axs[0, 1].set_title(dataset_name + ': $f$ vs $\max(c_x, c_y)$')
    axs[0, 1].set_xlabel('$\max(c_x, c_y)$')
    axs[0, 1].set_ylabel('$f$')

    axs[1, 0].scatter(q_values_avg, focal_lengths, alpha=0.5, marker='.', s=3, color='b')
    axs[1, 0].set_title(dataset_name +': $f$ vs $\\frac{c_x + c_y}{2}$')
    axs[1, 0].set_xlabel('$\\frac{c_x + c_y}{2}$')
    axs[1, 0].set_ylabel('$f$')

    axs[1, 1].scatter(q_values_sqrt, focal_lengths, alpha=0.5, marker='.', s=3, color='m')
    axs[1, 1].set_title(dataset_name + ': $f$ vs $\\sqrt{c_x c_y}$')
    axs[1, 1].set_xlabel('$\\sqrt{c_x c_y}$')
    axs[1, 1].set_ylabel('$f$')

    plt.tight_layout()
    fig_name = os.path.join(ppath, dataset_name.lower() + '_focal_length_vs_parameter.pdf')
    plt.savefig(fig_name, dpi = fig_dpi, bbox_inches='tight')

def plot_focal_length_vs_parameter_combine(datasets, dataset_names):
    ppath = os.path.join(bench_path, 'res', 'latex')
    os.makedirs(ppath, exist_ok=True)
    fig_dpi = 300

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    colors = ['b', 'g', 'r']

    for data, dataset_name, color in zip(datasets, dataset_names, colors):
        focal_lengths = []
        q_values_min = []
        q_values_max = []
        q_values_avg = []
        q_values_sqrt = []

        for key, K in data.items():
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            q_min = min(cx, cy)
            q_max = max(cx, cy)
            q_avg = (cx + cy) / 2
            q_sqrt = np.sqrt(cx * cy)

            focal_length = max(fx, fy)

            focal_lengths.append(focal_length)
            q_values_min.append(q_min)
            q_values_max.append(q_max)
            q_values_avg.append(q_avg)
            q_values_sqrt.append(q_sqrt)

        axs[0, 0].scatter(q_values_min, focal_lengths, alpha=0.2, marker='.', s=3, color=color, label=dataset_name)
        axs[0, 0].set_title('$f$ vs $\min(c_x, c_y)$')
        axs[0, 0].set_xlabel('$\min(c_x, c_y)$')
        axs[0, 0].set_ylabel('$f$')

        axs[0, 1].scatter(q_values_max, focal_lengths, alpha=0.2, marker='.', s=3, color=color, label=dataset_name)
        axs[0, 1].set_title('$f$ vs $\max(c_x, c_y)$')
        axs[0, 1].set_xlabel('$\max(c_x, c_y)$')
        axs[0, 1].set_ylabel('Focal Length')

        axs[1, 0].scatter(q_values_avg, focal_lengths, alpha=0.2, marker='.', s=3, color=color, label=dataset_name)
        axs[1, 0].set_title('$f$ vs $\\frac{c_x + c_y}{2}$')
        axs[1, 0].set_xlabel('$\\frac{c_x + c_y}{2}$')
        axs[1, 0].set_ylabel('$f$')

        axs[1, 1].scatter(q_values_sqrt, focal_lengths, alpha=0.2, marker='.', s=3, color=color, label=dataset_name)
        axs[1, 1].set_title('$f$ vs $\\sqrt{c_x c_y}$')
        axs[1, 1].set_xlabel('$\\sqrt{c_x c_y}$')
        axs[1, 1].set_ylabel('$f$')

    for ax in axs.flat:
        ax.legend()

    plt.tight_layout()
    fig_name = os.path.join(ppath, 'focal_length_vs_all_parameter.pdf')
    plt.savefig(fig_name, dpi = fig_dpi, bbox_inches='tight')


def plot_focal_length_vs_parameter_separate(datasets, dataset_names):
    ppath = os.path.join(bench_path, 'res', 'latex')
    os.makedirs(ppath, exist_ok=True)
    fig_dpi = 300

    colors = ['b', 'g', 'r']
    markers = ['x', '+', 'o']

    # plt.figure()
    # for data, dataset_name, color in zip(datasets, dataset_names, colors):
    #     focal_lengths = []
    #     q_values_min = []

    #     for key, K in data.items():
    #         fx = K[0, 0]
    #         fy = K[1, 1]
    #         cx = K[0, 2]
    #         cy = K[1, 2]

    #         q_min = min(cx, cy)
    #         focal_length = max(fx, fy)

    #         focal_lengths.append(focal_length)
    #         q_values_min.append(q_min)

    #     plt.scatter(q_values_min, focal_lengths, alpha=0.2, marker='.', s=3, color=color, label=dataset_name)

    # plt.title('$f$ vs $\min(c_x, c_y)$')
    # plt.xlabel('$\min(c_x, c_y)$')
    # plt.ylabel('$f$')
    # plt.legend()
    # plt.tight_layout()
    # fig_name = os.path.join(ppath, 'focal_length_vs_min_cx_cy.pdf')
    # plt.savefig(fig_name, dpi = fig_dpi, bbox_inches='tight')

    fig, ax = plt.subplots()
    for data, dataset_name, color, marker in zip(datasets, dataset_names, colors, markers):
        focal_lengths = []
        q_values_max = []

        for key, K in data.items():
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            q_max = max(cx, cy)
            focal_length = max(fx, fy)

            focal_lengths.append(focal_length)
            q_values_max.append(q_max)

        plt.scatter(q_values_max, focal_lengths, alpha=0.2, marker=marker, s=15, color=color, label=dataset_name)

    # plt.title('$f$ vs $\max(c_x, c_y)$')
    plt.xlabel('$\max(c_x, c_y)$')
    plt.ylabel('$f$')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.plot([1, 850], [1, 850 *2], 'k:', alpha=0.5, label = '$f = 2 \max(c_x, c_y)$')
    plt.legend()
    plt.tight_layout()
    fig_name = os.path.join(ppath, 'focal_length_vs_max_cx_cy.pdf')
    plt.savefig(fig_name, dpi = fig_dpi, bbox_inches='tight')

    # plt.figure()
    # for data, dataset_name, color in zip(datasets, dataset_names, colors):
    #     focal_lengths = []
    #     q_values_avg = []

    #     for key, K in data.items():
    #         fx = K[0, 0]
    #         fy = K[1, 1]
    #         cx = K[0, 2]
    #         cy = K[1, 2]

    #         q_avg = (cx + cy) / 2
    #         focal_length = max(fx, fy)

    #         focal_lengths.append(focal_length)
    #         q_values_avg.append(q_avg)

    #     plt.scatter(q_values_avg, focal_lengths, alpha=0.2, marker='.', s=3, color=color, label=dataset_name)

    # plt.title('$f$ vs $\\frac{c_x + c_y}{2}$')
    # plt.xlabel('$\\frac{c_x + c_y}{2}$')
    # plt.ylabel('$f$')
    # plt.legend()
    # plt.tight_layout()
    # fig_name = os.path.join(ppath, 'focal_length_vs_half_cx_cy.pdf')
    # plt.savefig(fig_name, dpi = fig_dpi, bbox_inches='tight')

    # plt.figure()
    # for data, dataset_name, color in zip(datasets, dataset_names, colors):
    #     focal_lengths = []
    #     q_values_sqrt = []

    #     for key, K in data.items():
    #         fx = K[0, 0]
    #         fy = K[1, 1]
    #         cx = K[0, 2]
    #         cy = K[1, 2]

    #         q_sqrt = np.sqrt(cx * cy)
    #         focal_length = max(fx, fy)

    #         focal_lengths.append(focal_length)
    #         q_values_sqrt.append(q_sqrt)

    #     plt.scatter(q_values_sqrt, focal_lengths, alpha=0.2, marker='.', s=3, color=color, label=dataset_name)

    # plt.title('$f$ vs $\sqrt{c_x c_y}$')
    # plt.xlabel('$\sqrt{c_x c_y}$')
    # plt.ylabel('$f$')
    # plt.legend()
    # plt.tight_layout()
    # fig_name = os.path.join(ppath, 'focal_length_vs_sqrt_cx_cy.pdf')
    # plt.savefig(fig_name, dpi = fig_dpi, bbox_inches='tight')
    

if __name__ == '__main__':
    bench_path = '../bench_data' 
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": "Times",
        })

    megadepth_data = magadepth_intrinsics_statistics(bench_path)
    scannet_data = scannet_intrinsics_statistics(bench_path)
    phototourism_data = phototourism_intrinsics_statistics(bench_path)

    # plot_focal_length_vs_parameter(bench_path, megadepth_data, "MegaDepth")
    # plot_focal_length_vs_parameter(bench_path, scannet_data, "ScanNet")
    # plot_focal_length_vs_parameter(bench_path, phototourism_data, "PhotoTourism")

    datasets = [phototourism_data, megadepth_data, scannet_data]
    dataset_names = ["IMC PhotoTourism", "MegaDepth", "ScanNet"]

    # plot_focal_length_vs_parameter_combine(datasets, dataset_names)
    plot_focal_length_vs_parameter_separate(datasets, dataset_names)
