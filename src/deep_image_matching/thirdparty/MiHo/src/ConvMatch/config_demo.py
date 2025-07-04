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
    "--layer_num", type=int, default=6, help=""
    "number of layers. Default: 6")
net_arg.add_argument(
    "--grid_num", type=int, default=16, help=""
    "number of grids in one domention. Default: 16")
net_arg.add_argument(
    "--net_channels", type=int, default=128, help=""
    "number of channels in a layer. Default: 128")
net_arg.add_argument(
    "--head", type=int, default=4, help=""
    "number of head in attention. Default: 4")
net_arg.add_argument(
    "--use_fundamental", type=str2bool, default=False, help=""
    "train fundamental matrix estimation. Default: False")
net_arg.add_argument(
    "--use_ratio", type=int, default=0, help=""
    "use ratio test. 0: not use, 1: use before network, 2: use as side information. Default: 0")
net_arg.add_argument(
    "--use_mutual", type=int, default=0, help=""
    "use matual nearest neighbor check. 0: not use, 1: use before network, 2: use as side information. Default: 0")
net_arg.add_argument(
    "--inlier_threshold", type=float, default=2.0, help=""
    "inlier threshold. Default: 2.0")

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def print_usage():
    parser.print_usage()

#
# config.py ends here
