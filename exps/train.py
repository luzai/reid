from __future__ import print_function, absolute_import
import argparse
import os.path as osp, yaml
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/xinglu/prj/open-reid/')

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss, TupletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

import torchvision
from tensorboardX import SummaryWriter
import lz


def get_data(name, split_id, data_dir, height, width, batch_size, num_instances=None,
             workers=32, combine_trainval=True):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])
    if num_instances is not None:
        train_loader = DataLoader(
            Preprocessor(train_set, root=dataset.images_dir,
                         transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            sampler=RandomIdentitySampler(train_set, num_instances),
            pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(
            Preprocessor(train_set, root=dataset.images_dir,
                         transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader


def main(args):
    lz.logging.info('config is {}'.format(vars(args)))
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    cudnn.benchmark = True
    writer = SummaryWriter(args.logs_dir)

    # Redirect print to both console and log file
    # sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    if args.num_instances is not None:
        assert args.batch_size % args.num_instances == 0, \
            'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
            (256, 128)
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.workers,
                 args.combine_trainval)
    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)

    model = models.create(args.arch,
                          num_features=args.features,
                          dropout=args.dropout,
                          branchs=args.branchs,
                          branch_dim=args.branch_dim,
                          use_global=args.use_global,
                          normalize=args.normalize)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    if len(args.gpu) == 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = nn.DataParallel(model.cuda(args.gpu[0]), device_ids=args.gpu, output_device=args.gpu[0])

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model)

    if hasattr(model.module, 'base') and args.freeze:
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)

    def adjust_lr(epoch, optimizer=optimizer, base_lr=args.lr, steps=args.steps):
        exp = len(steps)
        for i, step in enumerate(steps):
            if epoch < step:
                exp = i
                break
        lr = base_lr * 0.1 ** exp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Criterion # Optimizer

    if args.loss == 'triplet':
        criterion = TripletLoss(margin=args.margin, mode=args.mode).cuda(args.gpu[0])
    elif args.loss == 'tuple':
        criterion = TupletLoss(margin=args.margin).cuda(args.gpu[0])
    elif args.loss == 'softmax':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu[0])
    else:
        raise NotImplementedError

    # Trainer
    trainer = Trainer(model, criterion)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch=epoch)
        hist = trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq)
        for k, v in hist.iteritems():
            writer.add_scalar('train/' + k, v, epoch)
        if epoch < args.start_save:
            continue
        if epoch < args.epochs // 2 and epoch % 10 != 0:
            continue
        elif epoch < args.epochs - 20 and epoch % 5 != 0:
            continue

        acc = evaluator.evaluate(val_loader, dataset.val, dataset.val, metric, return_all=True)
        acc = {'top-1': acc['cuhk03'][0],
               'top-5': acc['cuhk03'][4],
               'top-10': acc['cuhk03'][9]
               }
        writer.add_scalars('train', acc, epoch)

        # if args.combine_trainval:
        acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric, return_all=True)
        # else:
        #     top1 = evaluator.evaluate(val_loader, dataset.val, dataset.val, return_all=True)
        acc = {'top-1': acc['cuhk03'][0],
               'top-5': acc['cuhk03'][4],
               'top-10': acc['cuhk03'][9]
               }
        writer.add_scalars('test', acc, epoch)

        top1 = acc['top-1']

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.{}.pth'.format(epoch)))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth'))
    model.module.load_state_dict(checkpoint['state_dict'])
    metric.train(model, train_loader)
    acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric, final=True)
    lz.logging.info('final rank1 is {}'.format(acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="many kind loss classification")

    # data
    parser.add_argument('-j', '--workers', type=int, default=32)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation",
                        default=True)
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")

    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of all parameters")
    parser.add_argument('--steps', type=list, default=[100, 150, 180])

    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=5)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])

    # misc
    home_dir = osp.expanduser('~') + '/.torch/'
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(home_dir, 'data'))

    # tuning

    configs_str = '''
        
    # - dataset: cuhk03
    #   logs_dir: logs.resnet152
    #   batch_size: 300
    #   gpu: [0,1]

    # - dataset: cuhk03
    #   arch: resnet50
    #   dropout: 0
    #   logs_dir: logs.bs320
    #   batch_size: 320
    #   gpu: [2,3]  
    
    # - dataset: cuhk03
    #   logs_dir: logs.bs480
    #   batch_size: 480
    #   gpu: [0,1,2]       
    
    - arch: resnet50 
      dropout: 0 
      logs_dir: logs.res.0.sgd
      gpu: [1,]

    # - arch: resnet50
    #   dropout: 0.3 
    #   logs_dir: logs.res.0.3 
    
    # - arch: attention50 
    #   gpu: [0,]
    #   branchs: 8
    #   branch_dim: 64
    #   dropout: 0.3
    #   freeze: False
    #   use_global: False
    #   normalize: True
    #   logs_dir: logs.at.8.64.0.3    

    # - arch: attention50 
    #   branchs: 8
    #   branch_dim: 128
    #   dropout: 0.3
    #   logs_dir: logs.at.8.128.0.3

    # - dataset: cuhk03
    #   arch: attention50
    #   logs_dir: logs.at.dp.0.3
    # 
    # - arch: attention50
    #   branchs: 8
    #   branch_dim: 64
    #   dropout: 0.8
    #   logs_dir: logs.at.0.8
    
    # - dataset: cuhk03 
    #   mode: lift
    #   logs_dir: logs.tri.lift
    
    '''
    parser.add_argument('--freeze', action='store_true', default=False)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--branchs', type=int, default=8)
    parser.add_argument('--branch_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--use_global', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=True)

    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--num_classes', type=int, default=128)
    parser.add_argument('--start_save', type=int, default=180,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=160)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../works/logs'))

    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--loss', type=str, default='triplet',
                        choices=['triplet', 'tuple', 'softmax'])
    parser.add_argument('--mode', type=str, default='hard',
                        choices=['rand', 'hard', 'all', 'lift'])
    parser.add_argument('--gpu', type=list, default=[0, ])
    dbg = True

    args = parser.parse_args()

    if args.loss == 'softmax':
        args.num_instances = None

    for config in yaml.load(configs_str):
        lz.logging.info('training {}'.format(config))
        for k, v in config.iteritems():
            if k not in vars(args):
                raise ValueError('{} {}'.format(k, v))
            setattr(args, k, v)
        args.logs_dir = '../work/' + args.logs_dir
        # lz.get_dev(ok=(0, 1,))
        if dbg:
            args.gpu = [lz.get_dev(n=1)]
            args.epochs = 2
            args.workers = 32
            args.batch_size = 160
            args.logs_dir += '.dbg'
        if len(args.gpu) == 1:
            lz.init_dev(args.gpu)
        lz.mkdir_p(args.logs_dir, delete=True)
        lz.write_json(vars(args), args.logs_dir + '/conf.json')

        proc = lz.mp.Process(target=main, args=(args,))
        proc.start()
        proc.join()

        # main(args)
