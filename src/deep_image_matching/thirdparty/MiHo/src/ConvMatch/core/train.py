import torch
import torch.optim as optim
from tqdm import trange
import os
from valid import valid
from loss import MatchLoss
from utils import tocuda
from tensorboardX import SummaryWriter


def train_step(step, optimizer, model, match_loss, data):
    model.train()
    if step == 80000 + 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2.

    res_logits, res_e_hat = model(data)
    loss = 0
    loss_val = []
    for i in range(len(res_logits)):
        loss_i, geo_loss, cla_loss, l2_loss, _, _ = match_loss.run(step, data, res_logits[i], res_e_hat[i])
        loss += loss_i
        loss_val += [geo_loss, cla_loss, l2_loss]
    optimizer.zero_grad()
    loss.backward()
    # sun3d training
    # if step == 80000 + 1:
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
    for name, param in model.named_parameters():
        if torch.any(torch.isnan(param.grad)):
            print('skip because nan')
            return loss_val

    optimizer.step()
    return loss_val


def train(model, train_loader, valid_loader, config):
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr, weight_decay = config.weight_decay)
    match_loss = MatchLoss(config)

    checkpoint_path = os.path.join(config.log_path, 'checkpoint.pth')
    config.resume = os.path.isfile(checkpoint_path)
    writer=SummaryWriter(os.path.join(config.log_path, 'log_file'))
    if config.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        best_acc = -1
        start_epoch = 0
    train_loader_iter = iter(train_loader)
    for step in trange(start_epoch, config.train_iter, ncols=config.tqdm_width):
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)
        train_data = tocuda(train_data)

        # run training
        cur_lr = optimizer.param_groups[0]['lr']
        loss_vals = train_step(step, optimizer, model, match_loss, train_data)
        if step%config.log_intv==0:
            writer.add_scalar('lr', cur_lr, step)
            writer.add_scalar('ClassifyLoss', loss_vals[1], step)
            writer.add_scalar('RegressionLoss', loss_vals[2], step)

        # Check if we want to write validation
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0
        if b_validate:
            va_res, geo_loss, cla_loss, l2_loss,  _, _, _  = valid(valid_loader, model, step, config)
            writer.add_scalar('val_ClassifyLoss', cla_loss, step)
            writer.add_scalar('val_RegressionLoss', l2_loss, step)
            writer.add_scalar('val_acc', va_res, step)            
            if va_res > best_acc:
                print("Saving best model with va_res = {}".format(va_res))
                best_acc = va_res
                torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, os.path.join(config.log_path, 'model_best.pth'))

        if b_save:
            torch.save({
            'epoch': step + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, checkpoint_path)

