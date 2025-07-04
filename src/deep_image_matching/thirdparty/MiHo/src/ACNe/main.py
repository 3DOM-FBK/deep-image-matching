#!/usr/bin/env python3
# Filename: main.py
# License: LICENSES/LICENSE_UVIC_EPFL

from __future__ import print_function

from config import get_config, print_usage
from data import load_data
from network import MyNetwork

eps = 1e-10
use3d = False
config = None

config, unparsed = get_config()

print("-------------------------Deep Essential-------------------------")
print("Note: To combine datasets, use .")


def main(config):
    """The main function."""


    # Run propper mode
    if config.run_mode == "train":

        # Load data train and validation data
        data = {}
        data["train"] = load_data(config, "train")
        data["valid"] = load_data(config, "valid")

        # Initialize network
        mynet = MyNetwork(config)

        # Run train
        mynet.train(data)

    elif config.run_mode == "test":

        # Load validation and test data. Note that we also load validation to
        # visualize more than what we did for training. For training we choose
        # minimal reporting to not slow down.
        data = {}
        # data["valid"] = load_data(config, "valid")
        data["test"] = load_data(config, "test")
        # Initialize network
        mynet = MyNetwork(config)

        # Run testing
        mynet.test(data)


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)

#
# main.py ends here
