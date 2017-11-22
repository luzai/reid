import os
from collections import defaultdict
from time import time

import torch


class LrUpdater(object):

    @staticmethod
    def step(optimizer, base_lr, epoch, step):
        lr = base_lr * (0.1**(epoch // step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def multistep(optimizer, base_lr, epoch, steps):
        exp = len(steps)
        for i, step in enumerate(steps):
            if epoch < step:
                exp = i
                break
        lr = base_lr * 0.1**exp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = defaultdict(float)
        self.sum = defaultdict(float)
        self.avg = defaultdict(float)
        self.count = defaultdict(int)
        self.reset()

    def reset(self, keys=None):
        if isinstance(keys, str):
            keys = [keys]
        elif keys is None:
            keys = self.val.keys()
        for k in keys:
            self.val[k] = 0
            self.sum[k] = 0
            self.avg[k] = 0
            self.count[k] = 0

    def update(self, pairs, n=1):
        for k, v in pairs.items():
            self.val[k] = v
            self.sum[k] += v * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]


class Runner(object):

    def __init__(self, model, optimizer, batch_processor, pavi=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_processor = batch_processor
        self.pavi = pavi
        self.epoch = 1
        self.num_iters = 0

    def load_checkpoint(self, filename):
        return load_checkpoint(self.model, filename)

    def save_checkpoint(self, out_dir, filename_tmpl='epoch_{}.pth'):
        save_checkpoint(self.model, self.epoch, self.num_iters, out_dir)

    def update_lr(self, epoch, base_lr, lr_step):
        if isinstance(lr_step, int):
            LrUpdater.step(self.optimizer, base_lr, epoch, lr_step)
        elif isinstance(lr_step, list):
            LrUpdater.multistep(self.optimizer, base_lr, epoch, lr_step)
        else:
            raise ValueError('"lr_step" must be an integer or list')

    def run_epoch(self, data_loader, train_mode, log_interval=0, **kwargs):
        avg_meter = AverageMeter()
        start = time()
        for i, data_batch in enumerate(data_loader):
            # measure data loading time
            avg_meter.update({'data_time': time() - start})

            outputs = self.batch_processor(self.model, data_batch, train_mode,
                                           **kwargs)
            if train_mode:
                self.num_iters += 1

            for key in outputs['losses']:
                avg_meter.update({
                    key: outputs['losses'][key].data[0]
                }, outputs['num_samples'])

            if train_mode:
                self.optimizer.zero_grad()
                outputs['losses']['loss'].backward()
                self.optimizer.step()
                if log_interval > 0 and (i + 1) % log_interval == 0:
                    lr = self.optimizer.param_groups[-1]['lr']
                    self.log(
                        self.epoch,
                        i + 1,
                        data_loader,
                        avg_meter,
                        log_vars=outputs['log_vars'],
                        lr=lr)

            # measure elapsed time
            avg_meter.update({'batch_time': time() - start})
            start = time()

        if not train_mode:
            self.log(
                self.epoch,
                i + 1,
                data_loader,
                avg_meter,
                log_vars=outputs['log_vars'])

    def log(self, epoch, i, data_loader, avg_meter, log_vars=None, lr=None):
        if lr is not None:
            log_info = 'Epoch [{}][{}/{}]\tlr: {:.5f}\t'.format(
                epoch, i, len(data_loader), lr)
        else:
            log_info = 'Epoch(val) [{}]\t'.format(epoch)

        log_info += ('Time {avg[batch_time]:.3f} (Data {avg[data_time]:.3f})\t'
                     'Loss {avg[loss]:.4f}').format(avg=avg_meter.avg)
        if log_vars is not None:
            assert isinstance(log_vars, (tuple, list))
            loss_items = [
                '{}: {:.4f}'.format(var, avg_meter.avg[var])
                for var in log_vars
            ]
            log_info += ' (' + ', '.join(loss_items) + ')'
        print(log_info)

        loss_names = ['loss'] + [var for var in log_vars]
        if self.pavi:
            phase = 'val' if lr is None else 'train'
            self.pavi.log(phase, self.num_iters,
                          {key: avg_meter.avg[key]
                           for key in loss_names})
        avg_meter.reset(loss_names)

    def resume(self, checkpoint_file):
        checkpoint = load_checkpoint(self.model, checkpoint_file)
        self.epoch = checkpoint['epoch'] + 1
        self.num_iters = checkpoint['num_iters']

    def run(self, train_loader, val_loader, base_lr, lr_step, max_epoch,
            val_interval, log_interval, checkpoint_dir, **kwargs):
        start_epoch = self.epoch
        for epoch in range(start_epoch - 1, max_epoch + 1):
            self.epoch = epoch
            self.update_lr(epoch, base_lr, lr_step)
            self.model.train()
            self.run_epoch(
                train_loader,
                train_mode=True,
                log_interval=log_interval,
                **kwargs)
            if epoch % val_interval == 0:
                self.model.eval()
                self.run_epoch(val_loader, train_mode=False, **kwargs)
                save_checkpoint(self.model, self.epoch, self.num_iters,
                                checkpoint_dir)


def load_checkpoint(model, filename):
    if not (isinstance(filename, str) and os.path.isfile(filename)):
        raise IOError('{} is not a checkpoint file'.format(filename))
    checkpoint = torch.load(filename)
    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {
            k.lstrip('module.'): v
            for k, v in checkpoint['state_dict'].items()
        }
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return checkpoint


def save_checkpoint(model,
                    epoch,
                    num_iters,
                    out_dir,
                    filename_tmpl='epoch_{}.pth'):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    filename = os.path.join(out_dir, filename_tmpl.format(epoch))
    torch.save({
        'epoch': epoch,
        'num_iters': num_iters,
        'state_dict': model.state_dict()
    }, filename)
    link_filename = os.path.join(out_dir, 'latest.pth')
    if os.path.islink(link_filename):
        os.remove(link_filename)
    os.symlink(filename, link_filename)
