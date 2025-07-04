import argparse



def str2bool(v):
    return v.lower() in ("true", "1")


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument(
    "--net_depth", type=int, default=12, help=""
    "number of layers. Default: 12")
net_arg.add_argument(
    "--clusters", type=int, default=500, help=""
    "cluster number in OANet. Default: 500")
net_arg.add_argument(
    "--iter_num", type=int, default=1, help=""
    "iteration number in the iterative network. Default: 1")
net_arg.add_argument(
    "--net_channels", type=int, default=128, help="" #此位置设定了N值
    "number of channels in a layer. Default: 128")
net_arg.add_argument(
    "--use_fundamental", type=str2bool, default=False, help=""
    "train fundamental matrix estimation. Default: False")
net_arg.add_argument(
    "--share", type=str2bool, default=False, help=""
    "share the parameter in iterative network. Default: False")
net_arg.add_argument(
    "--use_ratio", type=int, default=0, help=""
    "use ratio test. 0: not use, 1: use before network, 2: use as side information. Default: 0")
net_arg.add_argument(
    "--use_mutual", type=int, default=0, help=""
    "use matual nearest neighbor check. 0: not use, 1: use before network, 2: use as side information. Default: 0")
net_arg.add_argument(
    "--ratio_test_th", type=float, default=0.8, help=""
    "ratio test threshold. Default: 0.8")

# -----------------------------------------------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument(
    "--data_tr", type=str, default='../data_dump/yfcc-sift-2000-train.hdf5', help=""
    "name of the dataset for train")
data_arg.add_argument(
    "--data_va", type=str, default='../data_dump/yfcc-sift-2000-val.hdf5', help=""
    "name of the dataset for valid")
data_arg.add_argument(
    "--data_te", type=str, default='../data_dump/yfcc-sift-2000-test.hdf5', help=""
    "name of the unseen dataset for test")


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
    "--weight_decay", type=float, default=0, help=""
    "l2 decay")
loss_arg.add_argument(
    "--momentum", type=float, default=0.9, help=""
    "momentum")
loss_arg.add_argument(
    "--loss_classif", type=float, default=1.0, help=""
    "weight of the classification loss")
loss_arg.add_argument(
    "--loss_essential", type=float, default=0.5, help=""
    "weight of the essential loss")
loss_arg.add_argument(
    "--loss_essential_init_iter", type=int, default=2, help=""
    "initial iterations to run only the classification loss")
loss_arg.add_argument(
    "--geo_loss_margin", type=float, default=0.1, help=""
    "clamping margin in geometry loss")

# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument(
    "--run_mode", type=str, default="train", help=""
    "run_mode")
train_arg.add_argument(
    "--train_lr", type=float, default=1e-3, help=""
    "learning rate")
train_arg.add_argument(
    "--train_batch_size", type=int, default=4, help=""
    "batch size")
train_arg.add_argument(
    "--gpu_id", type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
train_arg.add_argument(
    "--num_processor", type=int, default=0, help='numbers of used cpu')
train_arg.add_argument(
    "--train_iter", type=int, default=10, help=""
    "training iterations to perform")
train_arg.add_argument(
    "--log_base", type=str, default="./log/", help=""
    "save directory name inside results")
train_arg.add_argument(
    "--log_suffix", type=str, default="", help=""
    "suffix of log dir")
train_arg.add_argument(
    "--val_intv", type=int, default=10, help=""
    "validation interval")
train_arg.add_argument(
    "--save_intv", type=int, default=2, help=""
    "summary interval")

# -----------------------------------------------------------------------------
# Testing
test_arg = add_argument_group("Test")
test_arg.add_argument(
    "--use_ransac", type=str2bool, default=False, help=""
    "use ransac when testing?")
test_arg.add_argument(
    "--model_path", type=str, default="", help=""
    "saved best model path for test")
test_arg.add_argument(
    "--res_path", type=str, default="", help=""
    "path for saving results")


# -----------------------------------------------------------------------------
# Visualization
vis_arg = add_argument_group('Visualization')
vis_arg.add_argument(
    "--tqdm_width", type=int, default=79, help=""
    "width of the tqdm bar"
)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
