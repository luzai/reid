from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

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

import torch
import torchvision
from tensorboardX import SummaryWriter


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
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    cudnn.benchmark = True
    writer = SummaryWriter(args.logs_dir)

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

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
    # todo draw model
    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)
    if args.loss != 'softmax':
        model = models.create(args.arch, num_features=1024,
                              dropout=args.dropout, num_classes=args.features)
    else:
        model = models.create(args.arch,
                              num_features=args.features,
                              dropout=args.dropout, num_classes=num_classes)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        metric.train(model, train_loader)
        print("Validation:")
        evaluator.evaluate(val_loader, dataset.val, dataset.val, metric)
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)
        return

    # Criterion # Optimizer
    if args.loss == 'triplet':
        criterion = TripletLoss(margin=args.margin, mode=args.mode).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)

        # Schedule learning rate
        def adjust_lr(epoch, decay_epoch=100):
            lr = args.lr if epoch <= decay_epoch else \
                args.lr * (0.001 ** ((epoch - decay_epoch) / float(decay_epoch/2.)))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)
    elif args.loss == 'tuple':
        criterion = TupletLoss(margin=args.margin).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)

        # Schedule learning rate
        def adjust_lr(epoch):
            lr = args.lr if epoch <= 100 else \
                args.lr * (0.001 ** ((epoch - 100) / 50.0))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)
    elif args.loss == 'softmax':
        criterion = nn.CrossEntropyLoss().cuda()
        if hasattr(model.module, 'base'):
            base_param_ids = set(map(id, model.module.base.parameters()))
            new_params = [p for p in model.parameters() if
                          id(p) not in base_param_ids]
            param_groups = [
                {'params': model.module.base.parameters(), 'lr_mult': 0.1},
                {'params': new_params, 'lr_mult': 1.0}]
        else:
            param_groups = model.parameters()
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)

        def adjust_lr(epoch):
            step_size = 60 if args.arch == 'inception' else 40
            lr = args.lr * (0.1 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

    # Trainer
    trainer = Trainer(model, criterion)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
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
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)


if __name__ == '__main__':
    import lz

    # lz.init_dev((0,1,2,3,))
    lz.init_dev((0,))
    parser = argparse.ArgumentParser(description="many kind loss classification")
    # tuning
    parser.add_argument('-b', '--batch-size', type=int, default=100)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--epochs', type=int, default=150)

    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs.attention'))

    parser.add_argument('-a', '--arch', type=str, default='attention50',
                        choices=models.names())
    parser.add_argument('--loss', type=str, default='triplet',
                        choices=['triplet', 'tuple', 'softmax'])
    parser.add_argument('--mode', type=str, default='hard',
                        choices=['rand', 'hard']
                        )

    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-j', '--workers', type=int, default=32)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
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
    # model

    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)  # 0.5
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of all parameters")  # 0.1
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=5)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])

    # misc
    home_dir = osp.expanduser('~') + '/.torch/'
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(home_dir, 'data'))

    args = parser.parse_args()
    args.logs_dir += ('.' + args.loss)
    dbg = False
    if dbg:
        lz.init_dev((0,))
        args.epochs = 2
        args.workers = 4
        args.batch_size = 8
        args.logs_dir = args.logs_dir + '.dbg'
    lz.mkdir_p(args.logs_dir, delete=True)
    lz.write_json(vars(args), args.logs_dir + '/conf.json')
    for k, v in vars(args).iteritems():
        print('{}: {}'.format(k, v))
    if args.loss == 'softmax':
        args.num_instances = None
    main(args)
