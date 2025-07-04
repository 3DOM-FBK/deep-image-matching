import numpy as np
import torch
from torch import nn
import torch.optim as optim
from tqdm import trange, tqdm
import os
import sys

from datasets import collate_fn, CorrespondencesDataset
from config import get_config, print_usage
from utils import compute_pose_error, pose_auc, estimate_pose_norm_kpts, estimate_pose_from_E, tocuda
from logger import Logger
from loss import MatchLoss
from model import CLNet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def create_log_dir(opt):
    if opt.log_suffix == "":
        suffix = sys.argv[0]
    result_path = opt.log_base+'/'+suffix

    if not os.path.isdir(opt.log_base):
        os.makedirs(opt.log_base)
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if not os.path.isdir(result_path+'/train'):
        os.makedirs(result_path+'/train')
    if not os.path.isdir(result_path+'/valid'):
        os.makedirs(result_path+'/valid')
    if not os.path.isdir(result_path+'/test'):
        os.makedirs(result_path+'/test')
    if os.path.exists(result_path+'/config.th'):
        print('warning: will overwrite config file')
    torch.save(opt, result_path+'/config.th')
    # path for saving traning logs
    opt.log_path = result_path+'/train'

def train_step(step, optimizer, model, match_loss, data):
    xs = data['xs']
    ys = data['ys']

    logits, ys_ds, e_hat, y_hat = model(xs, ys)
    loss, ess_loss, classif_loss = match_loss.run(step, data, logits, ys_ds, e_hat, y_hat)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        is_pos = (ys_ds[-1] < 1e-4).type(ys_ds[-1].type())
        is_neg = (ys_ds[-1] >= 1e-4).type(ys_ds[-1].type())
        inlier_ratio = torch.sum(is_pos, dim=-1) / (torch.sum(is_pos, dim=-1) + torch.sum(is_neg, dim=-1))
        inlier_ratio = inlier_ratio.mean().item()

    return [ess_loss, classif_loss, inlier_ratio]


def train(model, train_loader, valid_loader, opt):
    optimizer = optim.Adam(model.parameters(), lr=opt.train_lr, weight_decay=opt.weight_decay)
    match_loss = MatchLoss(opt)

    checkpoint_path = os.path.join(opt.log_path, 'checkpoint.pth')
    opt.resume = os.path.isfile(checkpoint_path)
    if opt.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        logger_valid = Logger(os.path.join(opt.log_path, 'log_valid.txt'), title='clnet', resume=True)
        logger_train = Logger(os.path.join(opt.log_path, 'log_train.txt'), title='clnet', resume=True)

    else:
        best_acc = -1
        start_epoch = 0

        logger_train = Logger(os.path.join(opt.log_path, 'log_train.txt'), title='clnet')
        logger_train.set_names(['Learning Rate'] + ['Essential Loss', 'Classfi Loss', 'Inlier ratio'])
        logger_valid = Logger(os.path.join(opt.log_path, 'log_valid.txt'), title='clnet')
        logger_valid.set_names(['AUC5'] + ['AUC10', 'AUC20'])

    train_loader_iter = iter(train_loader)

    tbar = trange(start_epoch, opt.train_iter)

    for step in tbar:
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)

        train_data = tocuda(train_data)

        # run training
        cur_lr = optimizer.param_groups[0]['lr']

        try:
            loss_vals = train_step(step, optimizer, model, match_loss, train_data)
        except:
            print("Skip unstable step")
            continue

        tbar.set_description('Doing: {}/{}, LR: {}, E_loss: {}, Cls_loss: {}, Inlier_Ratio: {}'\
        .format(step, opt.train_iter, cur_lr, loss_vals[0], loss_vals[1], loss_vals[2]))

        if step % 100 == 0:
            logger_train.append([cur_lr] + loss_vals)

        # Check if we want to write validation
        b_save = ((step + 1) % opt.save_intv) == 0
        b_validate = ((step + 1) % opt.val_intv) == 0

        if b_validate:
            aucs5, aucs10, aucs20 = valid(valid_loader, model, opt)
            logger_valid.append([aucs5, aucs10, aucs20])

            va_res = aucs5
            if va_res > best_acc:
                print("Saving best model with va_res = {}".format(va_res))
                best_acc = va_res
                torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, os.path.join(opt.log_path, 'model_best.pth'))

            model.train()

        if b_save:
            torch.save({
            'epoch': step + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, checkpoint_path)

def valid(valid_loader, model, opt):
    model.eval()
    err_ts, err_Rs = [], []
    for idx, valid_data in enumerate(tqdm(valid_loader)):
        xs = valid_data['xs'].cuda()
        ys = valid_data['ys'].cuda()

        _, _, e_hat, y_hat = model(xs, ys)

        mkpts0 = xs.squeeze()[:, :2].cpu().detach().numpy()
        mkpts1 = xs.squeeze()[:, 2:].cpu().detach().numpy()

        mask = y_hat.squeeze().cpu().detach().numpy() < opt.thr
        mask_kp0 = mkpts0[mask]
        mask_kp1 = mkpts1[mask]

        if opt.use_ransac == True:
            file_name = '/aucs.txt'
            ret = estimate_pose_norm_kpts(mask_kp0, mask_kp1)
        else:
            file_name = '/aucs_DLT.txt'
            e_hat = e_hat[-1].view(3, 3).cpu().detach().numpy()

            ret = estimate_pose_from_E(mkpts0, mkpts1, mask, e_hat)

        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            R_gt, t_gt = valid_data['Rs'], valid_data['ts']
            T_0to1 = torch.cat([R_gt.squeeze(), t_gt.squeeze().unsqueeze(-1)], dim=-1).numpy()
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        err_ts.append(err_t)
        err_Rs.append(err_R)

    # Write the evaluation results to disk.
    out_eval = {'error_t': err_ts,
                'error_R': err_Rs,
                }

    pose_errors = []
    for idx in range(len(out_eval['error_t'])):
        pose_error = np.maximum(out_eval['error_t'][idx], out_eval['error_R'][idx])
        pose_errors.append(pose_error)

    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]

    print('Evaluation Results (mean over {} pairs):'.format(len(test_loader)))
    print('AUC@5\t AUC@10\t AUC@20\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs[0], aucs[1], aucs[2]))

    return aucs[0], aucs[1], aucs[2]

if __name__ == "__main__":
    # ----------------------------------------
    # Parse configuration
    opt, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    create_log_dir(opt)

    # Initialize network
    model = CLNet(opt)
    model = model.cuda()

    print("Loading training data")
    train_dataset  = CorrespondencesDataset(opt.data_tr, opt)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.train_batch_size, num_workers=opt.num_processor, pin_memory=True, collate_fn=collate_fn)

    print("Training set len:", len(train_loader)*opt.train_batch_size)

    test_dataset = CorrespondencesDataset(opt.data_te, opt)

    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_processor, pin_memory=True, collate_fn=collate_fn)

    train(model, train_loader, test_loader, opt)
