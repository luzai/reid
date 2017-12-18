import torch, sys

sys.path.insert(0, '/home/xinglu/prj/open-reid/')

from lz import *
import lz
from torch.backends import cudnn
from torch.utils.data import DataLoader
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

from tensorboardX import SummaryWriter
import torchpack


def run(_):
    cfgs = torchpack.load_cfg('./conf/trihard.py')
    procs = []
    for args in cfgs.cfgs:

        args.logs_dir = 'work/' + args.logs_dir
        if args.gpu is not None:
            args.gpu = lz.get_dev(n=len(args.gpu), ok=range(8), mem=[0.9, 0.9])

        if isinstance(args.gpu, int):
            args.gpu = [args.gpu]
        if args.export_config:
            lz.mypickle(args, './conf.pkl')
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

    if isinstance(name, list) and len(name) != 1:
        raise NotImplementedError
    else:
        root = osp.join(data_dir, name_val)
        dataset_val = datasets.create(name_val, root, split_id=split_id)
        if name_val == 'market1501':
            lim_query = cvb.load(work_path + '/mk.query.pkl')
            dataset_val.query = [ds for ds in dataset_val.query if ds[0] in lim_query]
            lim_gallery = cvb.load(work_path + '/mk.gallery.pkl')
            dataset_val.gallery = [ds for ds in dataset_val.gallery if ds[0] in lim_gallery + lim_query]

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
    else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RandomCropFlip(height, width),
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
        Preprocessor(dataset_val.val, root=dataset_val.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=pin_memory)
    query_ga = np.concatenate([
        np.asarray(dataset_val.query).reshape(-1, 3),
        np.asarray(list(set(dataset_val.gallery) - set(dataset_val.query))).reshape(-1, 3)
    ])
    query_ga = np.rec.fromarrays((query_ga[:, 0], query_ga[:, 1].astype(int), query_ga[:, 2].astype(int)),
                                 names=['fnames', 'pids', 'cids']).tolist()
    test_loader = DataLoader(
        Preprocessor(query_ga,
                     root=dataset_val.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=pin_memory)
    dataset.val = dataset_val.val
    dataset.query = dataset_val.query
    dataset.gallery = dataset_val.gallery
    dataset.images_dir = dataset_val.images_dir
    # dataset.num_val_ids
    return dataset, num_classes, train_loader, val_loader, test_loader


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
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.workers,
                 args.combine_trainval, pin_memory=args.pin_mem, name_val=args.dataset_val)
    # Create model
    base_model = models.create(args.arch,
                               dropout=args.dropout,
                               pretrained=args.pretrained,
                               num_features=args.global_dim,
                               num_classes=args.num_classes
                               )
    embed_model = EltwiseSubEmbed()
    model = TripletNet(base_model, embed_model)
    model = torch.nn.DataParallel(model).cuda()

    if args.retrain:
        checkpoint = load_checkpoint(args.retrain)
        copy_state_dict(checkpoint['state_dict'], base_model, strip='module.')
        copy_state_dict(checkpoint['state_dict'], embed_model, strip='module.')

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> start epoch {}  best top1 {:.1%}"
              .format(args.start_epoch, best_top1))
    else:
        best_top1 = 0

    # Evaluator
    evaluator = SiameseEvaluator(
        torch.nn.DataParallel(base_model).cuda(),
        torch.nn.DataParallel(embed_model).cuda())
    if args.evaluate:
        print("Validation:")
        evaluator.evaluate(val_loader, dataset.val, dataset.val)
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
        return

    if args.hard_examples:
        # Use sequential train set loader
        data_loader = DataLoader(
            Preprocessor(dataset.train, root=dataset.images_dir,
                         transform=val_loader.dataset.transform),
            batch_size=args.batch_size, num_workers=args.workers,
            shuffle=False, pin_memory=False)
        # Mine hard triplet examples, index of [(anchor, pos, neg), ...]
        triplets = mine_hard_triplets(torch.nn.DataParallel(base_model).cuda(),
                                      data_loader, margin=args.margin)
        print("Mined {} hard example triplets".format(len(triplets)))
        # Build a hard examples loader
        train_loader.sampler = SubsetRandomSampler(triplets)

    # Criterion
    criterion = torch.nn.MarginRankingLoss(margin=args.margin).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Trainer
    trainer = TripletTrainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr * (0.1 ** (epoch // 40))
        for g in optimizer.param_groups:
            g['lr'] = lr

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)

        top1 = evaluator.evaluate(val_loader, dataset.val, dataset.val)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training Inception Siamese Model")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=['cuhk03', 'market1501', 'viper'])
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=2)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--hard-examples', action='store_true')
    # model
    parser.add_argument('--depth', type=int, default=50,
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    # loss
    parser.add_argument('--margin', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--retrain', type=str, default='', metavar='PATH')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
