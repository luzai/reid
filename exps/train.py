import torch
import lz
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import os.path as osp
import sys, yaml

sys.path.insert(0, '/home/xinglu/prj/open-reid/')
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from reid import datasets
from reid import models
from reid.models import *
from reid.dist_metric import DistanceMetric
from reid.loss import *
from reid.trainers import *
from reid.evaluators import *
from reid.mining import mine_hard_pairs
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import *
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

import torchvision
from tensorboardX import SummaryWriter


def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
             workers, combine_trainval, return_vis=False, pin_memory=False):
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

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=pin_memory, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=pin_memory)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=pin_memory)
    if not return_vis:
        return dataset, num_classes, train_loader, val_loader, test_loader
    else:
        return dataset, num_classes, train_loader, val_loader, test_loader, DataLoader(
            Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                         root=dataset.images_dir, transform=T.Compose([
                    T.RectScale(height, width),
                    T.ToTensor(),
                ])),
            batch_size=batch_size, num_workers=workers,
            shuffle=False, pin_memory=pin_memory
        )
def load_state_dict(model, state_dict):
    own_state = model.state_dict()
    success=[]
    for name, param in state_dict.items():
        if 'base_model.' + name in own_state:
            name = 'base_model.' + name
        if 'module.' + name in own_state:
            name = 'module.' + name
        if name not in own_state:
            print('ignore key "{}" in his state_dict'.format(name))
            continue

        if isinstance(param, nn.Parameter):
            param = param.data

        if own_state[name].size() == param.size():
            own_state[name].copy_(param)
            print('{} {} is ok '.format(name, param.size()))
            success.append(name )
        else:
            lz.logging.error('dimension mismatch for param "{}", in the model are {}'
                             ' and in the checkpoint are {}, ...'.format(
                name, own_state[name].size(), param.size()))

    missing = set(own_state.keys()) - set(success)
    if len(missing) > 0:
        print('missing keys in my state_dict: "{}"'.format(missing))

def main(args):
    # torch.cuda.set_device(args.gpu[0])
    lz.init_dev(args.gpu)
    lz.logging.info('config is {}'.format(vars(args)))
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    cudnn.benchmark = True
    writer = SummaryWriter(args.logs_dir)

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
            (256, 128)
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.workers,
                 args.combine_trainval, pin_memory=args.pin_mem)
    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=args.num_classes)
    print(model)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        # model.load_state_dict(checkpoint['state_dict'])
        load_state_dict(model, checkpoint['state_dict'])
        if args.restart:
            start_epoch_ = checkpoint['epoch']
            best_top1_ = checkpoint['best_top1']
            print("=> Start epoch {}  best top1 {:.1%}"
                  .format(start_epoch_, best_top1_))
        else:
            start_epoch = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
            print("=> Start epoch {}  best top1 {:.1%}"
                  .format(start_epoch, best_top1))
    # model = model.cuda()
    if len(args.gpu) == 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = nn.DataParallel(model, device_ids=range(len(args.gpu))).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric, final=True)
        lz.logging.info('final rank1 is {}'.format(acc))
        return 0

    # Criterion
    criterion = TripletLoss(margin=args.margin).cuda()

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay, momentum=0.9,
                                    nesterov=True)
    else:
        raise NotImplementedError
    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch, optimizer=optimizer, base_lr=args.lr, steps=args.steps, decay=args.decay):
        exp = len(steps)
        for i, step in enumerate(steps):
            if epoch < step:
                exp = i
                break
        lr = base_lr * decay ** exp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch=epoch)
        hist = trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq)
        for k, v in hist.items():
            writer.add_scalar('train/' + k, v, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

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
    import lz

    parser = argparse.ArgumentParser(description="many kind loss classification")
    parser.add_argument('--evaluate', action='store_true', default=False)
    # data
    parser.add_argument('--restart', action='store_true', default=True)
    parser.add_argument('--workers', type=int, default=4)
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
    parser.add_argument('--num-instances', type=int, default=4)

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
    parser.add_argument('--pin_mem', action="store_true", default=True)
    parser.add_argument('--decay', type=float, default=0.5)
    parser.add_argument('--config', metavar='PATH', default=None)
    parser.add_argument('--export-config', action='store_true', default=False)

    configs_str = '''    
    - arch: resnet50
            
      dataset: cuhk03
      workers: 4
      resume: '../work/logs.resnet50/model_best.pth'
      # resume: ''
      
      restart: True
      evaluate: False
      
      optimizer: sgd
      normalize: False
      dropout: 0 
      features: 512
      num_classes: 128 
      lr: 0.02
      decay: 0.5
      steps: [100,150,160 ]
      start_save: 0
      epochs: 180
      logs_dir: logs.resnet50.2
      batch_size: 120
      gpu: [3]
      pin_mem: True
    '''

    dbg = False

    args = parser.parse_args()

    for config in yaml.load(configs_str):
        for k, v in config.items():
            if k not in vars(args):
                raise ValueError('{} {}'.format(k, v))
            setattr(args, k, v)
        args.logs_dir = '../work/' + args.logs_dir
        if dbg:
            args.gpu = [lz.get_dev(n=1, mem=(0.5, 0.7))]
            args.epochs = 150
            args.workers = 4
            args.batch_size = 32
            args.logs_dir += '.dbg'

        if args.export_config:
            lz.mypickle((args), './conf.pkl')
            exit(0)

        lz.mkdir_p(args.logs_dir, delete=True)
        lz.write_json(vars(args), args.logs_dir + '/conf.json')

        proc = lz.mp.Process(target=main, args=(args,))
        proc.start()
        proc.join()

        # main(args)
