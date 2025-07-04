# Filename: data.py
# License: LICENSES/LICENSE_UVIC_EPFL
from __future__ import print_function

import os
import pickle

import h5py
import numpy as np

import cv2
from transformations import quaternion_from_matrix
from utils import loadh5, norm_points, norm_points_with_T, compute_T_with_imagesize



def load_geom(geom_file, geom_type, scale_factor, flip_R=False):
    if geom_type == "calibration":
        # load geometry file
        geom_dict = loadh5(geom_file)
        # Check if principal point is at the center
        K = geom_dict["K"]
        # assert(abs(K[0, 2]) < 1e-3 and abs(K[1, 2]) < 1e-3)
        # Rescale calbration according to previous resizing
        S = np.asarray([[scale_factor, 0, 0],
                        [0, scale_factor, 0],
                        [0, 0, 1]])
        K = np.dot(S, K)
        geom_dict["K"] = K
        # Transpose Rotation Matrix if needed
        if flip_R:
            R = geom_dict["R"].T.copy()
            geom_dict["R"] = R
        # append things to list
        geom_list = []
        geom_info_name_list = ["K", "R", "T", "imsize"]
        for geom_info_name in geom_info_name_list:
            geom_list += [geom_dict[geom_info_name].flatten()]
        # Finally do K_inv since inverting K is tricky with theano
        geom_list += [np.linalg.inv(geom_dict["K"]).flatten()]
        # Get the quaternion from Rotation matrices as well
        q = quaternion_from_matrix(geom_dict["R"])
        geom_list += [q.flatten()]
        # Also add the inverse of the quaternion
        q_inv = q.copy()
        np.negative(q_inv[1:], q_inv[1:])
        geom_list += [q_inv.flatten()]
        # Add to list
        geom = np.concatenate(geom_list)

    elif geom_type == "homography":
        H = np.loadtxt(geom_file)
        geom = H.flatten()

    return geom


def loadFromDir(train_data_dir, gt_div_str="", bUseColorImage=True,
                input_width=512, crop_center=True, load_lift=False):
    """Loads data from directory.

    train_data_dir : Directory containing data

    gt_div_str : suffix for depth (e.g. -8x8)

    bUseColorImage : whether to use color or gray (default false)

    input_width : input image rescaling size

    """

    # read the list of imgs and the homography
    train_data_dir = train_data_dir.rstrip("/") + "/"
    img_list_file = train_data_dir + "images.txt"
    geom_list_file = train_data_dir + "calibration.txt"
    vis_list_file = train_data_dir + "visibility.txt"
    depth_list_file = train_data_dir + "depth" + gt_div_str + ".txt"
    # parse the file
    image_fullpath_list = []
    with open(img_list_file, "r") as img_list:
        while True:
            # read a single line
            tmp = img_list.readline()
            if type(tmp) != str:
                line2parse = tmp.decode("utf-8")
            else:
                line2parse = tmp
            if not line2parse:
                break
            # strip the newline at the end and add to list with full path
            image_fullpath_list += [train_data_dir +
                                    line2parse.rstrip("\n")]
    # parse the file
    geom_fullpath_list = []
    with open(geom_list_file, "r") as geom_list:
        while True:
            # read a single line
            tmp = geom_list.readline()
            if type(tmp) != str:
                line2parse = tmp.decode("utf-8")
            else:
                line2parse = tmp
            if not line2parse:
                break
            # strip the newline at the end and add to list with full path
            geom_fullpath_list += [train_data_dir +
                                   line2parse.rstrip("\n")]

    # parse the file
    vis_fullpath_list = []
    with open(vis_list_file, "r") as vis_list:
        while True:
            # read a single line
            tmp = vis_list.readline()
            if type(tmp) != str:
                line2parse = tmp.decode("utf-8")
            else:
                line2parse = tmp
            if not line2parse:
                break
            # strip the newline at the end and add to list with full path
            vis_fullpath_list += [train_data_dir + line2parse.rstrip("\n")]

    # parse the file
    if os.path.exists(depth_list_file):
        depth_fullpath_list = []
        with open(depth_list_file, "r") as depth_list:
            while True:
                # read a single line
                tmp = depth_list.readline()
                if type(tmp) != str:
                    line2parse = tmp.decode("utf-8")
                else:
                    line2parse = tmp
                if not line2parse:
                    break
                # strip the newline at the end and add to list with full
                # path
                depth_fullpath_list += [train_data_dir +
                                        line2parse.rstrip("\n")]
    else:
        print("no depth file at {}".format(depth_list_file))
        # import IPython
        # IPython.embed()
        # exit
        depth_fullpath_list = [None] * len(vis_fullpath_list)

    # For each image and geom file in the list, read the image onto
    # memory. We may later on want to simply save it to a hdf5 file
    x = []
    geom = []
    vis = []
    depth = []
    kp = []
    desc = []
    idxImg = 1
    for img_file, geom_file, vis_file, depth_file in zip(
            image_fullpath_list, geom_fullpath_list, vis_fullpath_list,
            depth_fullpath_list):

        print('\r -- Loading Image {} / {}'.format(
            idxImg, len(image_fullpath_list)
        ), end="")
        idxImg += 1

        # ---------------------------------------------------------------------
        # Read the color image
        if not bUseColorImage:
            # If there is not gray image, load the color one and convert to
            # gray
            if os.path.exists(img_file.replace(
                    "image_color", "image_gray"
            )):
                img = cv2.imread(img_file.replace(
                    "image_color", "image_gray"
                ), 0)
                assert len(img.shape) == 2
            else:
                # read the image
                img = cv2.cvtColor(cv2.imread(img_file),
                                   cv2.COLOR_BGR2GRAY)
            if len(img.shape) == 2:
                img = img[..., None]
            in_dim = 1

        else:
            img = cv2.imread(img_file)
            in_dim = 3
        assert(img.shape[-1] == in_dim)

        # Crop center and resize image into something reasonable
        if crop_center:
            rows, cols = img.shape[:2]
            if rows > cols:
                cut = (rows - cols) // 2
                img_cropped = img[cut:cut + cols, :]
            else:
                cut = (cols - rows) // 2
                img_cropped = img[:, cut:cut + rows]
            scale_factor = float(input_width) / float(img_cropped.shape[0])
            img = cv2.resize(img_cropped, (input_width, input_width))
        else:
            scale_factor = 1.0

        # Add to the list
        x += [img.transpose(2, 0, 1)]

        # ---------------------------------------------------------------------
        # Read the geometric information in homography
        geom += [load_geom(
            geom_file,
            "calibration",
            scale_factor,
        )]

        # ---------------------------------------------------------------------
        # Load visibility
        vis += [np.loadtxt(vis_file).flatten().astype("float32")]

        # ---------------------------------------------------------------------
        # Load Depth
        depth += []             # Completely disabled
        # if depth_file is not None:
        #     cur_depth = loadh5(depth_file)["z"].T.astype("float32")
        #     # crop center
        #     if crop_center:
        #         if rows > cols:
        #             cut = (rows - cols) // 2
        #             depth_cropped = cur_depth[cut:cut + cols, :]
        #         else:
        #             cut = (cols - rows) // 2
        #             depth_cropped = cur_depth[:, cut:cut + rows]
        #         # resize
        #         depth_resized = cv2.resize(
        #             depth_cropped, (input_width, input_width))
        #         depth += [depth_resized.reshape([1, input_width, input_width])]
        #     else:
        #         depth += [cur_depth[None]]
        # else:
        #     # raise RuntimeError("No depth file!")
        #     # depth += [-1e6 * np.ones((1, input_width, input_width))]
        #     depth += []

        # TODO: Load keypoints and descriptors from the precomputed files here.
        #
        # NOTE: Use the last element added to get the geom and depth
        #
        if load_lift:
            desc_file = img_file + ".desc.h5"
            with h5py.File(desc_file, "r") as ifp:
                h5_kp = ifp["keypoints"].value[:, :2]
                h5_desc = ifp["descriptors"].value
            # Get K (first 9 numbers of geom)
            K = geom[-1][:9].reshape(3, 3)
            # Get cx, cy
            h, w = x[-1].shape[1:]
            cx = (w - 1.0) * 0.5
            cy = (h - 1.0) * 0.5
            cx += K[0, 2]
            cy += K[1, 2]
            # Get focals
            fx = K[0, 0]
            fy = K[1, 1]
            # New kp
            kp += [
                (h5_kp - np.array([[cx, cy]])) / np.asarray([[fx, fy]])
            ]
            # New desc
            desc += [h5_desc]

    print("")

    return (x, np.asarray(geom),
            np.asarray(vis), depth, kp, desc)


def load_data(config, var_mode):
    """Main data loading routine"""

    # insert other dataset format such oan
    if config.data_name.startswith("oan"):
        print("load {}".format(config.data_name))
        if var_mode == "train":
            """
            Since training set from OANet is too large. We use the data_loader from OANet repo. 
            """
            import multiprocessing as mp
            from data_loader_oanet import CorrespondencesDataset, collate_fn
            import torch.utils.data
            num_core = int(mp.cpu_count()) - 3
            train_dataset = CorrespondencesDataset(config.data_name, config, "train")
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config.train_batch_size, shuffle=True,
                num_workers=num_core, pin_memory=False, collate_fn=collate_fn)
            return train_loader
        else:
            data = load_data_oan(config, var_mode)
    else:
        print("Loading {} data".format(var_mode))

        # use only the first two characters for shorter abbrv
        var_mode = var_mode[:2]

        # Now load data.
        var_name_list = [
            "xs", "ys", "Rs", "ts",
            "img1s", "cx1s", "cy1s", "f1s",
            "img2s", "cx2s", "cy2s", "f2s",
        ]
        

        data_folder = config.data_dump_prefix
        if config.use_lift:
            data_folder += "_lift"
        if config.use_lfnet:
            data_folder += "_lfnet"
        if config.use_sp:
            data_folder += "_sp"

        # Let's unpickle and save data
        data = {}
        if config.prefiltering is not "" or config.run_mode == "comp":
            var_name_list += ["mutuals", "ratios"]
        else:
            data["mutuals"] = [] 
            data["ratios"] = []

        data_names = getattr(config, "data_" + var_mode)
        data_names = data_names.split(".")
        for data_name in data_names:
            cur_data_folder = "/".join([
                data_folder,
                data_name,
                "numkp-{}".format(config.obj_num_kp),
                "nn-{}".format(config.obj_num_nn),
            ])
            if not config.data_crop_center:
                cur_data_folder = os.path.join(cur_data_folder, "nocrop")
            suffix = "{}-{}".format(
                var_mode,
                getattr(config, "train_max_" + var_mode + "_sample")
            )
            cur_folder = os.path.join(cur_data_folder, suffix)
            ready_file = os.path.join(cur_folder, "ready")
            if not os.path.exists(ready_file):
                # data_gen_lock.unlock()
                raise RuntimeError("Data is not prepared!")

            appendix = ""

            for var_name in var_name_list:
                cur_var_name = var_name + "_" + appendix + var_mode
                in_file_name = os.path.join(cur_folder, cur_var_name) + ".pkl"
                with open(in_file_name, "rb") as ifp:
                    if var_name in data:
                        data[var_name] += pickle.load(ifp)
                    else:
                        data[var_name] = pickle.load(ifp)
                print("{} loaded!".format(in_file_name))
    
    mutuals = data["mutuals"]
    ratios = data["ratios"]
    xs = data["xs"]
    ys = data["ys"]
    ratio_test = 0.8
    if config.prefiltering == "R":
        for i in range(len(xs)):
            _x = xs[i]
            _y = ys[i]
            _mask = ratios[i] < ratio_test 
            _x = _x.squeeze(0)[_mask][None]
            _y = _y[_mask]
            xs[i] = _x
            ys[i] = _y
    elif config.prefiltering == "B":
        for i in range(len(xs)):
            _x = xs[i]
            _y = ys[i]
            _mask = mutuals[i].astype(bool)
            _x = _x.squeeze(0)[_mask][None]
            _y = _y[_mask]
            xs[i] = _x
            ys[i] = _y
    elif config.prefiltering == "RB":
        for i in range(len(xs)):
            _x = xs[i]
            _y = ys[i]
            _mask_ratio = ratios[i] < ratio_test
            _mask_matching = mutuals[i].astype(bool)
            _mask = np.all([_mask_matching, _mask_ratio], axis=0)
            _x = _x.squeeze(0)[_mask][None]
            _y = _y[_mask]
            xs[i] = _x
            ys[i] = _y
    elif config.prefiltering == "":
        print("No prefiltering on dataset")
    else:
        raise ValueError("Wrong prefiltering type!")
    
    data["xs"] = xs
    data["ys"] = ys

    if config.use_fundamental == 0:
        data["T1s"] = []
        data["T2s"] = []
        data["K1s"] = []
        data["K2s"] = []
    elif config.use_fundamental > 0:
        # go back pixel coordinates and normalize with image size
        xs = data["xs"]
        ys = data["ys"]
        Rs = data["Rs"]
        ts = data["ts"]
        cx1s = data["cx1s"]
        cy1s = data["cy1s"]
        f1s = data["f1s"]
        cx2s = data["cx2s"]
        cy2s = data["cy2s"]
        f2s = data["f2s"]
        data["T1s"] = []
        data["T2s"] = []
        data["K1s"] = []
        data["K2s"] = []

        # calculating average f_gt / f_image_size
        ratio = 1.0 

        for i in range(len(xs)):
            x_cur = xs[i]
            x1, x2 = x_cur[0, :, :2], x_cur[0, :, 2:4]
            cx1 = np.asarray(cx1s[i]).squeeze()
            cy1 = np.asarray(cy1s[i]).squeeze()
            cx2 = np.asarray(cx2s[i]).squeeze()
            cy2 = np.asarray(cy2s[i]).squeeze()
            f1 = np.asarray(f1s[i]).squeeze()
            f2 = np.asarray(f2s[i]).squeeze()

            # in case single f
            if f1.size == 2:
                f1i = f1[0]
                f1j = f1[1]
            else:
                f1i = f1
                f1j = f1

            if f2.size == 2:
                f2i = f2[0]
                f2j = f2[1]
            else:
                f2i = f2
                f2j = f2
                
            K1 = np.array([
                [f1i, 0, cx1],
                [0, f1j, cy1],
                [0, 0, 1]
                ])
            K2 = np.array([
                [f2i, 0, cx2],
                [0, f2j, cy2],
                [0, 0, 1]
                ])

            # move back pixel coordinates and then normalize points
            x1 = x1 * np.asarray([K1[0,0], K1[1,1]]) + np.array([K1[0,2], K1[1,2]])
            x2 = x2 * np.asarray([K2[0,0], K2[1,1]]) + np.array([K2[0,2], K2[1,2]])
            # in CNe/OANet dataset, the [cx, cy] is the image center
            if config.use_fundamental == 2:
                w1 = cx1 * 2 + 1.0
                h1 = cy1 * 2 + 1.0 
                T1 = compute_T_with_imagesize(w1, h1, ratio=ratio)
                w2 = cx2 * 2 + 1.0
                h2 = cy2 * 2 + 1.0 
                T2 = compute_T_with_imagesize(w2, h2, ratio=ratio)
                x1 = norm_points_with_T(x1, T1)
                x2 = norm_points_with_T(x2, T2)
            elif config.use_fundamental == 1:
                x1, T1 = norm_points(x1)
                x2, T2 = norm_points(x2)
            else:
                raise ValueError("wrong Fundamental matrix")
                
            data["T1s"] += [T1]
            data["T2s"] += [T2]
            data["K1s"] += [K1]
            data["K2s"] += [K2]
            data["xs"][i] = np.concatenate([x1, x2], -1)[None]
    else:
        raise ValueError("wrong Fundamental matrix")


    return data


def load_data_oan(config, var_mode):
    """Main data loading routine"""

    print("Loading {} data".format(var_mode))

    # use only the first two characters for shorter abbrv
    var_mode = var_mode[:2]

    if config.data_name in ["oan_outdoor"]:
        data_dir = "data_dump_oan"
        filenames = {
            "tr": "yfcc-sift-2000-train.hdf5",
            "va": "yfcc-sift-2000-val.hdf5",
            "te": "yfcc-sift-2000-test.hdf5",
        }
    elif config.data_name in ["oan_indoor"]:
        data_dir = "data_dump_oan"
        filenames = {
            "tr": "sun3d-sift-2000-train.hdf5",
            "va": "sun3d-sift-2000-val.hdf5",
            "te": "sun3d-sift-2000-test.hdf5"}
    else:
        raise ValueError("wrong data_name")
    filename = os.path.join(data_dir, filenames[var_mode])
    print("loading {}".format(filename))
    data_dict = {}
    var_name_list = [
        "xs", "ys", "Rs", "ts",
        "img1s", "cx1s", "cy1s", "f1s",
        "img2s", "cx2s", "cy2s", "f2s",
        "mutuals", "ratios"
    ]
    for var_name in var_name_list:
        data_dict[var_name] = []
    with h5py.File(filename, 'r') as h5file:
        for key in h5file.keys():
            # fix the index
            for index in h5file[key].keys():
                index_ = str(index)
                if key == "ys":
                    # To be compatible with the format of CNe dataset.
                    # where ys has dimension Nkp*2.
                    v_ = np.array(h5file[key][index_])
                    if v_.shape[-1] == 1:
                        v_ = np.repeat(v_, 2, axis=-1)
                    data_dict[key] += [v_]
                else:
                    data_dict[key] += [np.array(h5file[key][index_])]
    return data_dict    
# ends here