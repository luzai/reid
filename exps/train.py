import torch, sys

sys.path.insert(0, '/home/xinglu/prj/open-reid/')

from lz import *
import lz
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
# from reid.mining import mine_hard_pairs
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import *
from reid.utils.logging import Logger
from reid.utils.serialization import *

import torchvision
from tensorboardX import SummaryWriter
import torchpack


def run(_):
    cfgs = torchpack.load_cfg('./conf/trihard.py')
    procs = []
    for args in cfgs.cfgs:

        args.logs_dir = 'work/' + args.logs_dir
        args.gpu = lz.get_dev(n=len(args.gpu), ok=range(8), mem=[0.9, 0.9])
        # args.gpu = [3,]

        if isinstance(args.gpu, int):
            args.gpu = [args.gpu]
        if args.export_config:
            lz.mypickle((args), './conf.pkl')
            exit(0)
        if not args.evaluate:
            assert args.logs_dir != args.resume
            lz.mkdir_p(args.logs_dir, delete=True)
            cvb.dump(args, args.logs_dir + '/conf.pkl')

        main(args)

        # proc = lz.mp.Process(target=main, args=(args,))
        # proc.start()
        # # time.sleep(30)
        # procs.append(proc)
        # # for proc in procs:
        # proc.join()


def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
             workers, combine_trainval, pin_memory=True, name_val=''):
    if isinstance(name, list) and len(name) != 1:
        names = name
        root = '/home/xinglu/.torch/data/'
        roots = [root + name_ for name_ in names]
        dataset = datasets.creates(name, roots=roots)
    else:
        root = osp.join(data_dir, name)
        dataset = datasets.create(name, root, split_id=split_id)

    if name_val != '':
        assert isinstance(name_val, str)
        root = osp.join(data_dir, name_val)
        dataset_val = datasets.create(name_val, root, split_id=split_id)
        dataset.qurey = dataset_val.query
        dataset.gallery = dataset_val.gallery

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
        sampler=RandomIdentityWeightedSampler(train_set, num_instances, batch_size=batch_size),
        pin_memory=pin_memory, drop_last=True)

    fnames = np.asarray(train_set)[:, 0]
    fname2ind = dict(zip(fnames, np.arange(fnames.shape[0])))
    setattr(train_loader, 'fname2ind', fname2ind)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=pin_memory)
    query_ga = np.concatenate([
        np.asarray(dataset.query).reshape(-1, 3),
        np.asarray(list(set(dataset.gallery) - set(dataset.query))).reshape(-1, 3)
    ])
    query_ga = np.rec.fromarrays((query_ga[:, 0], query_ga[:, 1].astype(int), query_ga[:, 2].astype(int)),
                                 names=['fnames', 'pids', 'cids']).tolist()
    test_loader = DataLoader(
        Preprocessor(query_ga,
                     root='/home/xinglu/.torch/data/market1501/images/',  # dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=pin_memory)

    # query, gallery = dataset.query.copy(), dataset.gallery.copy()
    # query_ids = [pid for _, pid, _ in query]
    # gallery_ids = [pid for _, pid, _ in gallery]
    # query_ids_u = np.unique(query_ids)
    # residual = np.setdiff1d(np.unique(gallery_ids), query_ids_u)
    # limit = np.random.choice(query_ids_u, min(100, query_ids_u.shape[0]), replace=False)
    # limit = np.concatenate((residual, limit))
    # query = limit_dataset(query, limit)
    # gallery = limit_dataset(gallery, limit)

    # query_ga = np.concatenate([
    #     np.asarray(query).reshape(-1, 3),
    #     np.asarray(list(set(gallery) - set(query))).reshape(-1, 3)
    # ])
    # query_ga = np.rec.fromarrays((query_ga[:, 0], query_ga[:, 1].astype(int), query_ga[:, 2].astype(int)),
    #                              names=['fnames', 'pids', 'cids']).tolist()
    # test_loader_limit = DataLoader(
    #     Preprocessor(query_ga,
    #                  root=dataset.images_dir,
    #                  transform=test_transformer),
    #     batch_size=batch_size, num_workers=workers,
    #     shuffle=False, pin_memory=pin_memory)
    ## todo not support vis anymore
    return dataset, num_classes, train_loader, val_loader, test_loader


def limit_dataset(query, limit):
    res_l = []
    for pid_, df_ in pd.DataFrame(data=query, columns=['fnames', 'pids', 'cids']).groupby('pids'):
        if pid_ not in limit: continue
        res_l.append(df_)
    res = pd.concat(res_l)
    return res.to_records(index=False).tolist()


def main(args):
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    sys.stderr = Logger(osp.join(args.logs_dir, 'err.txt'))
    lz.init_dev(args.gpu)
    print('config is {}'.format(vars(args)))
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    cudnn.benchmark = True
    if not args.evaluate:
        writer = SummaryWriter(args.logs_dir)

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
                 args.combine_trainval, pin_memory=args.pin_mem, name_val=args.dataset_val)
    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)

    # base_model = models.create(args.arch,
    #                            dropout=args.dropout,
    #                            pretrained=args.pretrained,
    #                            cut_at_pooling=True
    #                            )
    # if args.branchs * args.branch_dim != 0:
    #     local_model = Mask(2048, args.branchs, args.branch_dim,
    #                        height=args.height // 32,
    #                        width=args.width // 32,
    #                        dopout=args.dropout
    #                        )
    # else:
    #     local_model = None
    # if args.global_dim != 0:
    #     global_model = Global(2048, args.global_dim, dropout=args.dropout)
    # else:
    #     global_model = None
    # concat_model = ConcatReduce(args.branchs * args.branch_dim + args.global_dim,
    #                             args.num_classes,dropout=0)
    #
    # model = SingleNet(base_model, global_model, local_model, concat_model)

    model = models.create(args.arch,
                          pretrained=args.pretrained,
                          dropout=args.dropout,
                          norm=args.normalize,
                          num_features=args.global_dim,
                          num_classes=args.num_classes
                          )

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
    evaluator = Evaluator(model)
    if args.evaluate:
        acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric, final=True)
        lz.logging.info('eval cmc-1 is {}'.format(acc))
        # db = lz.Database('distmat.h5', 'a')
        # db['ohnm'] = evaluator.distmat
        # db.close()
        return 0

    # Criterion
    criterion = TripletLoss(margin=args.margin).cuda()

    # Optimizer
    # filter(lambda p: p.requires_grad, model.parameters())
    for param in itertools.chain(model.module.base.parameters(),
                                 model.module.feat.parameters(),
                                 model.module.feat_bn.parameters(),
                                 model.module.classifier.parameters()
                                 ):
        param.requires_grad = False

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.module.stn.parameters(), lr=args.lr,  # module.stn
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay, momentum=0.9,
                                    nesterov=True)
    else:
        raise NotImplementedError
    # Trainer
    trainer = Trainer(model, criterion, dbg=True, logs_at=args.logs_dir + '/vis')

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
        if not args.log_middle:
            continue
        if epoch < args.start_save:
            continue
        if epoch not in args.log_at:
            continue

        mAP, acc = evaluator.evaluate(val_loader, dataset.val, dataset.val, metric)
        writer.add_scalar('train/top-1', acc, epoch)
        writer.add_scalar('train/mAP', mAP, epoch)

        # mAP, acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)
        # writer.add_scalar('test/top-1', acc, epoch)
        # writer.add_scalar('test/mAP', mAP, epoch)

        top1 = acc
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
        mAP, acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric, final=True)
        writer.add_scalar('test/top-1', acc, args.epochs + 1)
        writer.add_scalar('test/mAP', mAP, args.epochs + 1)
        lz.logging.info('final rank1 is {}'.format(acc))


if __name__ == '__main__':
    run('')
