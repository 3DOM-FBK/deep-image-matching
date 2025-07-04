# Filename: config.py
# License: LICENSES/LICENSE_UVIC_EPFL

import argparse

from .servers import is_computecanada, is_cvlab_epfl, is_vcg_uvic


def str2bool(v):
    return v.lower() in ("true", "1")


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# -----------------------------------------------------------------------------
# NeighNet
neigh_arg = add_argument_group("NeighNet")
neigh_arg.add_argument(
    "--num_inner", type=int, default=2, help=""
    "num of inner block in each residual block")

neigh_arg.add_argument(
    "--num_acn_inner", type=int, default=1, help=""
    "num of acn in each inner block")

neigh_arg.add_argument(
    "--use_fundamental", type=int, default=0, help=""
    "0: essential matrix case, 1: fundamental under image-size norm")

neigh_arg.add_argument(
    "--prefiltering", type=str, default="", help=""
    "do prefiltering on dataset")

neigh_arg.add_argument(
    "--save_test_dir", type=str, default="", help=""
    "save test dir")

neigh_arg.add_argument(
    "--F_th", type=float, default=-1, help=""
    "threshold")
neigh_arg.add_argument(
    "--E_th", type=float, default=-1, help=""
    "threshold")

neigh_arg.add_argument(
    "--bool_hard_attention", type=str2bool, default=False, help=""
    "button to trun on the hard attention"
)

neigh_arg.add_argument(
    "--matching_crossCheck", type=str2bool, default=False, help=""
    "whether do the cross check when dumping the data"
)

neigh_arg.add_argument(
    "--bool_use_weight_for_score", type=str2bool, default=False, help=""
    "button to trun on the hard attention"
)
neigh_arg.add_argument(
    "--nonlinearity_output", type=str2bool, default=False, help=""
    "whether add linearity output before sigmoid")
neigh_arg.add_argument(
    "--ratio_test", type=float, default=None, help=""
    "do ratio test during feature matching")
neigh_arg.add_argument(
    "--sigma", type=float, default=1.0, help=""
    "number of layers")
neigh_arg.add_argument(
    "--loss_gaussian_sigma", type=float, default=1.0, help=""
    "number of layers")
neigh_arg.add_argument(
    "--loss_var", type=float, default=0.0, help=""
    "number of layers")
neigh_arg.add_argument(
    "--loss_multi_logit", type=float, default=0.0, help=""
    "number of layers")
neigh_arg.add_argument(
    "--loss_multi_logit_ada", type=float, default=0.0, help=""
    "number of layers")
neigh_arg.add_argument(
    "--loss_multi_logit_aux", type=float, default=0.0, help=""
    "number of layers")
neigh_arg.add_argument(
    "--loss_hinge", type=float, default=0.0, help=""
    "number of layers")
neigh_arg.add_argument(
    "--th_logit", type=float, default=0.0, help=""
    "threshold for logit")
neigh_arg.add_argument(
    "--loss_err", type=float, default=0.0, help=""
    "number of layers")
neigh_arg.add_argument(
    "--loss_attention", type=float, default=0.0, help=""
    "number of layers")
neigh_arg.add_argument(
    "--loss_inner", type=float, default=0.0, help=""
    "number of layers")
neigh_arg.add_argument(
    "--n_group", type=int, default=1, help=""
    "number of groups")
neigh_arg.add_argument(
    "--num_phase", type=int, default=1, help=""
    "number of groups")
neigh_arg.add_argument(
    "--gcn_opt", type=str, default="vanilla", help=""
    "where the activation should be in case of resnet")
neigh_arg.add_argument(
    "--bn_opt", type=str, default="bn", help=""
    "where the activation should be in case of resnet")
neigh_arg.add_argument(
    "--divergence_opt", type=str, default="cross_entropy", help=""
    "where the activation should be in case of resnet")
neigh_arg.add_argument(
    "--weight_opt", type=str, default="relu_tanh", help=""
    "where the activation should be in case of resnet")
neigh_arg.add_argument(
    "--gcn_pos", type=str, default="pre", help=""
    "where the activation should be in case of resnet")

neigh_arg.add_argument(
    "--w_opt", type=str, default=None,
    choices=["none", None, "l2", "l2_sigma", "9w"], help=""
    "where the activation should be in case of resnet")
neigh_arg.add_argument(
    "--kernel_size", type=float, default=0.0, help=""
    "kernel_size for the 9w strategy")

neigh_arg.add_argument(
    "--pre_x_in", type=str, default="4",
    choices=["4", "3", "9"], help=""
    "Different choice for the input")
neigh_arg.add_argument(
    "--var_mode", type=str, default="division", help=""
    "Different choice for the input")
neigh_arg.add_argument(
    "--mean_mode", type=str, default="1", help=""
    "calculate the mean value for the gcn layer")

# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument(
    "--net_depth", type=int, default=12, help=""
    "number of layers")
net_arg.add_argument(
    "--net_nchannel", type=int, default=128, help=""
    "number of channels in a layer")
net_arg.add_argument(
    "--net_act_pos", type=str, default="post",
    choices=["pre", "mid", "post"], help=""
    "where the activation should be in case of resnet")
net_arg.add_argument(
    "--net_gcnorm", type=str2bool, default=True, help=""
    "whether to use context normalization for each layer")
net_arg.add_argument(
    "--net_batchnorm", type=str2bool, default=True, help=""
    "whether to use batch normalization")
net_arg.add_argument(
    "--net_bn_test_is_training", type=str2bool, default=False, help=""
    "is_training value for testing")

# -----------------------------------------------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument(
    "--data_dump_prefix", type=str, default="./data_dump", help=""
    "prefix for the dump folder locations")
data_arg.add_argument(
    "--data_name", type=str, default="cne", help=""
    "prefix for the dump folder locations")
data_arg.add_argument(
    "--data_tr", type=str, default="st_peters", help=""
    "name of the dataset for train")
data_arg.add_argument(
    "--data_va", type=str, default="st_peters", help=""
    "name of the dataset for valid")
data_arg.add_argument(
    "--data_te", type=str, default="st_peters", help=""
    "name of the dataset for test")
data_arg.add_argument(
    "--data_crop_center", type=str2bool, default=False, help=""
    "whether to crop center of the image "
    "to match the expected input for methods that expect a square input")
data_arg.add_argument(
    "--use_lift", type=str2bool, default=False, help=""
    "if this is set to true, we expect lift to be dumped already for all "
    "images.")
data_arg.add_argument(
    "--use_lfnet", type=str2bool, default=False, help=""
    "if this is set to true, we expect lift to be dumped already for all "
    "images.")
data_arg.add_argument(
    "--use_sp", type=str2bool, default=False, help=""
    "if this is set to true, we expect lift to be dumped already for all "
    "images.")


# -----------------------------------------------------------------------------
# Objective
obj_arg = add_argument_group("obj")
obj_arg.add_argument(
    "--obj_num_kp", type=int, default=2000, help=""
    "number of keypoints per image")
obj_arg.add_argument(
    "--obj_top_k", type=int, default=-1, help=""
    "number of keypoints above the threshold to use for "
    "essential matrix estimation. put -1 to use all. ")
obj_arg.add_argument(
    "--obj_num_nn", type=int, default=1, help=""
    "number of nearest neighbors in terms of descriptor "
    "distance that are considered when generating the "
    "distance matrix")
obj_arg.add_argument(
    "--obj_geod_type", type=str, default="episym",
    choices=["sampson", "episqr", "episym"], help=""
    "type of geodesic distance")
obj_arg.add_argument(
    "--obj_geod_th", type=float, default=1e-4, help=""
    "theshold for the good geodesic distance")

# -----------------------------------------------------------------------------
# Loss
loss_arg = add_argument_group("loss")
loss_arg.add_argument(
    "--loss_decay", type=float, default=0.0, help=""
    "l2 decay")
loss_arg.add_argument(
    "--loss_EM", type=float, default=0.0, help=""
    "Use EM loss")
loss_arg.add_argument(
    "--loss_gaussian", type=float, default=0.0, help=""
    "Use EM loss")
loss_arg.add_argument(
    "--loss_classif", type=float, default=1.0, help=""
    "weight of the classification loss")
loss_arg.add_argument(
    "--loss_essential", type=float, default=0.1, help=""
    "weight of the essential loss")
loss_arg.add_argument(
    "--loss_essential_init_iter", type=int, default=20000, help=""
    "initial iterations to run only the classification loss")

# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument(
    "--run_mode", type=str, default="train", help=""
    "run_mode")
train_arg.add_argument(
    "--train_batch_size", type=int, default=32, help=""
    "batch size")
train_arg.add_argument(
    "--train_max_tr_sample", type=int, default=10000, help=""
    "number of max training samples")
train_arg.add_argument(
    "--train_max_va_sample", type=int, default=1000, help=""
    "number of max validation samples")
train_arg.add_argument(
    "--train_max_te_sample", type=int, default=1000, help=""
    "number of max test samples")
train_arg.add_argument(
    "--train_lr", type=float, default=1e-3, help=""
    "learning rate")
train_arg.add_argument(
    "--train_iter", type=int, default=500000, help=""
    "training iterations to perform")
train_arg.add_argument(
    "--res_dir", type=str, default="./logs", help=""
    "base directory for results")
train_arg.add_argument(
    "--log_dir", type=str, default="", help=""
    "save directory name inside results")
train_arg.add_argument(
    "--test_log_dir", type=str, default="", help=""
    "which directory to test inside results")
train_arg.add_argument(
    "--val_intv", type=int, default=5000, help=""
    "validation interval")
train_arg.add_argument(
    "--report_intv", type=int, default=1000, help=""
    "summary interval")

# -----------------------------------------------------------------------------
# Visualization
vis_arg = add_argument_group('Visualization')
vis_arg.add_argument(
    "--vis_dump", type=str2bool, default=False, help=""
    "turn this on to dump data for visualization"
)
vis_arg.add_argument(
    "--tqdm_width", type=int, default=79, help=""
    "width of the tqdm bar"
)
vis_arg.add_argument(
    "--vis_dir", type=str, default="", help=""
    "vis is not none, then dump mask_before into vis_dir"
)



def setup_dataset(dataset_name):
    """Expands dataset name and directories properly"""

    # Use only the first one for dump
    dataset_name = dataset_name.split(".")[0]

    # Setup the base directory depending on the environment
    if is_computecanada():
        data_dir = "/scratch/kyi/datasets/scenedata_splits/"
    elif is_vcg_uvic():
        data_dir = "/data/datasets/sfm/scenedata_splits/"
    elif is_cvlab_epfl():
        data_dir = "/cvlabdata2/home/kyi/Datasets/scenedata_splits/"
    else:
        data_dir = "./datasets/"

    # Expand the abbreviations that we use to actual folder names
    if "cogsci4" == dataset_name:
        # Load the data
        data_dir += "brown_cogsci_4---brown_cogsci_4---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.3
    elif "reichstag" == dataset_name:
        # Load the data
        data_dir += "reichstag/"
        geom_type = "Calibration"
        vis_th = 100
    elif "sacre_coeur" == dataset_name:
        # Load the data
        data_dir += "sacre_coeur/"
        geom_type = "Calibration"
        vis_th = 100
    elif "buckingham" in dataset_name:
        # Load the data
        data_dir += "buckingham_palace/"
        geom_type = "Calibration"
        vis_th = 100
    elif "notre_dame" == dataset_name:
        # Load the data
        data_dir += "notre_dame_front_facade/"
        geom_type = "Calibration"
        vis_th = 100
    elif "st_peters" == dataset_name:
        # Load the data
        data_dir += "st_peters_square/"
        geom_type = "Calibration"
        vis_th = 100
    elif "harvard_conf_big" == dataset_name:
        # Load the data
        data_dir += "harvard_conf_big---hv_conf_big_1---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.3
    elif "home_ac" == dataset_name:
        # Load the data
        data_dir += "home_ac---home_ac_scan1_2012_aug_22---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.3
    elif "fountain" in dataset_name:
        # Load the data
        data_dir += "fountain/"
        geom_type = "Calibration"
        vis_th = -1
    elif "herzjesu" == dataset_name:
        # Load the data
        data_dir += "herzjesu/"
        geom_type = "Calibration"
        vis_th = -1
    elif "gms-teddy" == dataset_name:
        # Load the data
        data_dir += "gms-teddy/"
        geom_type = "Calibration"
        vis_th = 100
    elif "gms-large-cabinet" in dataset_name:
        # Load the data
        data_dir += "gms-large-cabinet/"
        geom_type = "Calibration"
        vis_th = 100
    elif "cogsci8_05" == dataset_name:
        # Load the data
        data_dir += "brown_cogsci_8---brown_cogsci_8---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "cogsci2_05" == dataset_name:
        # Load the data
        data_dir += "brown_cogsci_2---brown_cogsci_2---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_lounge1_2_05" == dataset_name:
        # Load the data
        data_dir += "harvard_corridor_lounge---hv_lounge1_2---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_c10_2_05" == dataset_name:
        # Load the data
        data_dir += "harvard_c10---hv_c10_2---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_s1_2_05" == dataset_name:
        # Load the data
        data_dir += "harvard_robotics_lab---hv_s1_2---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_c4_1_05" == dataset_name:
        # Load the data
        data_dir += "harvard_c4---hv_c4_1---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "cs7_05" == dataset_name:
        # Load the data
        data_dir += "brown_cs_7---brown_cs7---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "cs3_05" == dataset_name:
        # Load the data
        data_dir += "brown_cs_3---brown_cs3---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "mit_46_6conf_05" == dataset_name:
        # Load the data
        data_dir += "mit_46_6conf---bcs_floor6_conf_1---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "mit_46_6lounge_05" == dataset_name:
        # Load the data
        data_dir += "mit_46_6lounge---bcs_floor6_long---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "mit_w85g_05" == dataset_name:
        # Load the data
        data_dir += "mit_w85g---g_0---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "mit_32_g725_05" == dataset_name:
        # Load the data
        data_dir += "mit_32_g725---g725_1---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "florence_hotel_05" == dataset_name:
        # Load the data
        data_dir += "hotel_florence_jx---florence_hotel_stair_room_all---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "mit_w85h_05" == dataset_name:
        # Load the data
        data_dir += "mit_w85h---h2_1---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "cogsci6_05" == dataset_name:
        # Load the data
        data_dir += "brown_cogsci_6---brown_cogsci_6---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    # New sets
    elif "home_ac_05_fix" == dataset_name:
        # Load the data
        data_dir += "home_ac---home_ac_scan1_2012_aug_22---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "harvard_conf_big_05_fix" == dataset_name:
        # Load the data
        data_dir += "harvard_conf_big---hv_conf_big_1---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "cogsci3_05" == dataset_name:
        # Load the data
        data_dir += "brown_cogsci_3---brown_cogsci_3---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "cogsci4_05_fix" == dataset_name:
        # Load the data
        data_dir += "brown_cogsci_4---brown_cogsci_4---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "home_aca_05_fix" == dataset_name:
        # Load the data
        data_dir += "home_ag---apartment_ag_nov_7_2012_scan1_erika---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hotel_ucsd_05" == dataset_name:
        # Load the data
        data_dir += "hotel_ucsd---la2-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "brown_cs_4_05" == dataset_name:
        data_dir += "brown_cs_4---brown_cs4-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hotel_ucla_ant_05" == dataset_name:
        # Load the data
        data_dir += "hotel_ucla_ant---hotel_room_ucla_scan1_2012_oct_05-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_lounge3_05" == dataset_name:
        data_dir += "harvard_corridor_lounge---hv_lounge_corridor3_whole_floor-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "harvard_conf_big_05_rand" == dataset_name:
        # Load the data
        data_dir += "harvard_conf_big---hv_conf_big_1-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "brown_bm_3_05" == dataset_name:
        # Load the data
        data_dir += "brown_bm_3---brown_bm_3-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "home_pt_05" == dataset_name:
        # Load the data
        data_dir += "home_pt---home_pt_scan1_2012_oct_19-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_comp_05" == dataset_name:
        # Load the data
        data_dir += "harvard_computer_lab---hv_c1_1-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_lounge2_05" == dataset_name:
        # Load the data
        data_dir += "harvard_corridor_lounge---hv_lounge_corridor2_1-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hotel_ped_05" == dataset_name:
        # Load the data
        data_dir += "hotel_pedraza---hotel_room_pedraza_2012_nov_25-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5

    return data_dir, geom_type, vis_th


def get_config():
    config, unparsed = parser.parse_known_args()

    # Setup the dataset related things
    for _mode in ["tr", "va", "te"]:
        data_dir, geom_type, vis_th = setup_dataset(
            getattr(config, "data_" + _mode))
        setattr(config, "data_dir_" + _mode, data_dir)
        setattr(config, "data_geom_type_" + _mode, geom_type)
        setattr(config, "data_vis_th_" + _mode, vis_th)

    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
