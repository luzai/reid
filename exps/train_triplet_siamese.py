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


def get_data(name, split_id, data_dir, height, width, batch_size, num_instances=4,
             workers=32, combine_trainval=True, return_vis=False):
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
        pin_memory=True, drop_last=True)

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
            shuffle=False, pin_memory=True
        )


def main(args):
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
                 args.combine_trainval)

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)

    base_model = models.create(args.arch, pretrained=True,
                               cut_at_pooling=True,
                               dropout=args.dropout,
                               # norm= args.normalize,
                               # num_features= args.features ,
                               # num_classes=args.num_classes
                               )
    if args.embed == 'kron':
        embed_model = KronEmbed(8, 4, 128, 2)
    elif args.embed == 'concat':
        embed_model = ConcatEmbed(1024)
    elif args.embed == 'eltsub':
        EltwiseSubEmbed(args.features, args.num_classes)
    tranform = Transform(mode='hard')

    model = SiameseNet2(base_model, tranform, embed_model, )
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
    if len(args.gpu) == 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = nn.DataParallel(model, device_ids=range(len(args.gpu))).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = CascadeEvaluator(
        torch.nn.DataParallel(base_model).cuda(),
        torch.nn.DataParallel(embed_model).cuda(),
        embed_dist_fn=lambda x: F.softmax(Variable(x[:, 0]), dim=0)
    )
    if args.evaluate:
        acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, return_all=False)
        print('top-1 ', acc)
        return 0

    criterion = nn.CrossEntropyLoss().cuda()

    # base_param_ids = set(map(id, model.module.model.base.parameters()))
    # new_params = [p for p in model.parameters() if
    #               id(p) not in base_param_ids]
    # param_groups = [
    #     {'params': model.module.model.base.parameters(), 'lr_mult': 0.1},
    #     {'params': new_params, 'lr_mult': 1.0}]
    param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay,
                                nesterov=True)

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

    # Trainer
    trainer = VerfTrainer(model, criterion)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
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

        acc1, acc = evaluator.evaluate(val_loader, dataset.val, dataset.val, return_all=False)
        writer.add_scalars('train/top-1', {'stage1': acc1,
                                           'stage2': acc}, epoch)

        # if args.combine_trainval:
        acc1, acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, return_all=False)
        writer.add_scalars('test/top-1', {'stage1': acc1,
                                          'stage2': acc}, epoch)

        top1 = acc

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        # is_best = True
        # top1 = 0
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
    acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
    lz.logging.info('final rank1 is {}'.format(acc))


if __name__ == '__main__':
    import lz

    parser = argparse.ArgumentParser(description="many kind loss classification")
    # data
    parser.add_argument('--restart', action='store_true', default=True)
    parser.add_argument('-j', '--workers', type=int, default=4)
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
    # model
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)  # 0.5
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate of all parameters")
    parser.add_argument('--steps', type=list, default=[50, 100, 150])
    parser.add_argument('--epochs', type=int, default=160)

    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')  # ../work/logs.siamese/model_best.pth
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

    # tuning
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())

    parser.add_argument('-b', '--batch-size', type=int, default=64)
    working_dir = osp.dirname(osp.abspath(__file__))

    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs.dbg'))

    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--gpu', type=list, default=[3, ])
    parser.add_argument('--embed', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--decay', type=float,default=0.5)
    configs_str = '''
        - arch: resnet50
          dataset: cuhk03
          resume: '../work/logs.resnet50.2/model_best.pth'
          # resume: '../work/logs.siamese/model_best.pth'
          # resume: ''
          restart: True
          
          evaluate: False
          optimizer: sgd  
          
          embed: kron
          
          dropout: 0 
          lr: 0.02
          start_save: 0
          
          steps: [100,150,160]
          decay: 0.5
          epochs: 180
          
          logs_dir: logs.siamese.tri
          batch_size: 120
          gpu: [0,]
        '''

    dbg = False
    args = parser.parse_args()

    for config in yaml.load(configs_str):
        for k, v in config.items():
            if k not in vars(args):
                raise ValueError('{} {}'.format(k, v))
            setattr(args, k, v)
        args.logs_dir = '../work/' + args.logs_dir
        # lz.get_dev(ok=(0, 1,))
        if dbg:
            args.gpu = [0]
            args.epochs = 150
            args.workers = 4
            args.batch_size = 32
            args.logs_dir += '.dbg'
        lz.mkdir_p(args.logs_dir, delete=True)
        lz.write_json(vars(args), args.logs_dir + '/conf.json')

        proc = lz.mp.Process(target=main, args=(args,))
        proc.start()
        proc.join()

        # main(args)
