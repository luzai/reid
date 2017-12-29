import torch, sys

sys.path.insert(0, '/home/xinglu/prj/open-reid/')

from lz import *
import lz
import torch

# np.random.seed(1)
# torch.manual_seed(1)

from exps.opts import get_parser
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import reid
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
from reid.utils.serialization import *

import torchvision
from tensorboardX import SummaryWriter


def run(args):
    configs_str = '''        
        - arch: resnet50 
          print_freq: 1
          dataset: cuhk03
          optimizer: adam
          embed: concat
          mode: hard
          num_instances: 4
          # resume: 'work/siamese/model_best.pth'
          resume: 'work.long/fuck.all.data.long.cont3/model_best.pth'
          restart: True
          evaluate: False
          export_config: False
          dropout: 0 
          lr: 0.00005
          steps: [150,200,220]
          decay: 0.5
          epochs: 50
          freeze: ''
          logs_dir: work/siamese.contrastive
          start_save: 0
          log_start: False
          log_middle: True
          # log_at: [1,100,150,163,164]
          log_at: [1,164,] 
          need_second: True
          batch_size: 120
          gpu: [1, ]
          workers: 4
          margin: 0.25
        '''
    for config in yaml.load(configs_str):
        for k, v in config.items():
            if k not in vars(args):
                raise ValueError('{} {}'.format(k, v))
            setattr(args, k, v)
        args.logs_dir = args.logs_dir
        args.gpu = lz.get_dev(n=len(args.gpu), ok=(0, 1, 2, 3), mem=[0.5, 0.8])
        if isinstance(args.gpu, int):
            args.gpu = [args.gpu]
        if args.export_config:
            lz.mypickle((args), './conf.pkl')
            exit(0)
        if not args.evaluate:
            assert args.logs_dir != args.resume
            lz.mkdir_p(args.logs_dir, delete=True)

        proc = lz.mp.Process(target=main, args=(args,))
        proc.start()
        proc.join()

        # main(args)


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
        sampler=RandomIdentitySampler(train_set, num_instances, batch_size=batch_size),
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
    if not args.evaluate:
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
                               )
    if args.embed == 'kron':
        embed_model = KronEmbed(8, 4, 128, 2)
    elif args.embed == 'concat':
        embed_model = ConcatEmbed(4096)
    elif args.embed == 'eltsub':
        embed_model = EltwiseSubEmbed(args.features, args.num_classes)
    else:
        raise NotImplementedError
    tranform = Transform(mode=args.mode)

    model = SiameseNet3(base_model, tranform, embed_model)
    print(model)

    # Load from checkpoint
    if args.log_start:
        best_top1 = 0
        start_epoch = 1
    else:
        best_top1 = start_epoch = 0

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
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
        lz.logging.info('final rank1 is {}'.format(acc))
        db = lz.Database('distmat.h5', 'w')
        db['ohmn_match/1'] = evaluator.distmat1
        db['ohmn_match/2'] = evaluator.distmat2

        # acc = evaluator.evaluate(val_loader, dataset.val, dataset.val, return_all=False)
        # print('val rank1', acc)
        # db['ohmn_match/val/1'] = evaluator.distmat1
        # db['ohmn_match/val/2'] = evaluator.distmat2
        db.close()

        return 0

    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = reid.loss.ContrastiveLoss(margin=args.margin).cuda()

    # base_param_ids = set(map(id, model.module.model.base.parameters()))
    # new_params = [p for p in model.parameters() if
    #               id(p) not in base_param_ids]
    # param_groups = [
    #     {'params': model.module.model.base.parameters(), 'lr_mult': 0.1},
    #     {'params': new_params, 'lr_mult': 1.0}]
    # param_groups = model.parameters()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay, momentum=0.9,
                                    nesterov=True)
    else:
        raise NotImplementedError
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

    def train_log(epoch):
        acc1, acc = evaluator.evaluate(val_loader, dataset.val, dataset.val, return_all=False,
                                       need_second=args.need_second)
        writer.add_scalars('train/top-1', {'stage1': acc1,
                                           'stage2': acc}, epoch)

        # if args.combine_trainval:
        acc1, acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, return_all=False,
                                       need_second=args.need_second)
        writer.add_scalars('test/top-1', {'stage1': acc1,
                                          'stage2': acc}, epoch)

        return acc

    # Trainer
    trainer = VerfTrainer(model, criterion, )

    # Start training
    if args.log_start:
        train_log(0)
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        hist = trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq)
        for k, v in hist.items():
            writer.add_scalar('train/' + k, v, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        if not args.log_middle:
            continue
        if epoch < args.start_save:
            continue
        if epoch not in args.log_at:
            continue

        top1 = train_log(epoch)

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
    if osp.exists(osp.join(args.logs_dir, 'model_best.pth')):
        print('Test with best model:')
        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth'))
        model.module.load_state_dict(checkpoint['state_dict'])
        metric.train(model, train_loader)
        acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
        lz.logging.info('final rank1 is {}'.format(acc))


if __name__ == '__main__':
    parser = get_parser()

    args = parser.parse_args()
    run(args)
