# Code is heavily borrowed from https://github.com/zjhthu/OANet
# Author: Shihua Zhang
# Date: 2022/11/21
# E-mail: suhzhang001@gmail.com


from config import get_config, print_usage
config, unparsed = get_config()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
import torch.utils.data
import sys
from data import collate_fn, CorrespondencesDataset
from convmatch import ConvMatch as Model
from train import train


print("-------------------------Deep Essential-------------------------")
print("Note: To combine datasets, use .")

def create_log_dir(config):
    if not os.path.isdir(config.log_base):
        os.makedirs(config.log_base)
    if config.log_suffix == "":
        suffix = "-".join(sys.argv)
    else:
        suffix = config.log_suffix
    if not os.path.isdir(config.log_base):
        os.makedirs(config.log_base)
    result_path = config.log_base + suffix
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if not os.path.isdir(result_path+'/train'):
        os.makedirs(result_path+'/train')
    if not os.path.isdir(result_path+'/valid'):
        os.makedirs(result_path+'/valid')
    if os.path.exists(result_path+'/config.th'):
        print('warning: will overwrite config file')
    torch.save(config, result_path+'/config.th')
    # path for saving traning logs
    config.log_path = result_path#+'/train'

def main(config):
    """The main function."""

    # Initialize network
    model = Model(config)

    # Run propper mode
    create_log_dir(config)

    train_dataset = CorrespondencesDataset(config.data_tr, config)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.train_batch_size, shuffle=True,
            num_workers=16, pin_memory=False, collate_fn=collate_fn)

    valid_dataset = CorrespondencesDataset(config.data_va, config)
    valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=config.train_batch_size, shuffle=False,
            num_workers=8, pin_memory=False, collate_fn=collate_fn)
    #valid_loader = None
    print('start training .....')
    train(model, train_loader, valid_loader, config)


if __name__ == "__main__":
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)

#
# main.py ends here
