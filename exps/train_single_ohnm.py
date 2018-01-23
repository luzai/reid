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
from reid.mining import *
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import *
from reid.utils.logging import Logger
from reid.utils.serialization import *

from tensorboardX import SummaryWriter
import torchpack


def run(_):
    cfgs = torchpack.load_cfg('./cfgs/single_ohnm.py')
    procs = []
    for args in cfgs.cfgs:
        args.dbg = True
        if args.dbg:
            args.epochs = 1
            args.batch_size = 32

        if args.evaluate:
            args.logs_dir += '.eval'
        args.logs_dir = 'work/' + args.logs_dir
        if args.gpu is not None:
            # args.gpu = lz.get_dev(n=len(args.gpu), ok=(2,3), mem=[0.1, 0.1],sleep=22.33)
            args.gpu = lz.get_dev(n=len(args.gpu), ok=range(4), mem=[0.1, 0.4], sleep=10)
            args.gpu = (1,)

        if isinstance(args.gpu, int):
            args.gpu = [args.gpu]
        if not args.evaluate:
            assert args.logs_dir != args.resume
            lz.mkdir_p(args.logs_dir, delete=True)
            cvb.dump(args, args.logs_dir + '/conf.pkl')
        main(args)
        lz.mp.set_start_method('spawn')
        proc = lz.mp.Process(target=main, args=(args,))
        proc.start()
        time.sleep(12)
        procs.append(proc)

    for proc in procs:
        proc.join()


def get_data(args):
    name, split_id, data_dir, height, width, batch_size, num_instances, workers, combine_trainval = args.dataset, args.split, args.data_dir, args.height, args.width, args.batch_size, args.num_instances, args.workers, args.combine_trainval,
    pin_memory = args.pin_mem
    name_val = args.dataset_val
    npy = args.has_npy

    if isinstance(name, list) and len(name) != 1:
        names = name
        root = '/home/xinglu/.torch/data/'
        roots = [root + name_ for name_ in names]
        dataset = datasets.creates(name, roots=roots)
    else:
        root = osp.join(data_dir, name)
        dataset = datasets.create(name, root, split_id=split_id, mode=args.dataset_mode)

    if isinstance(name_val, list) and len(name_val) != 1:
        raise NotImplementedError
    else:
        root = osp.join(data_dir, name_val)
        dataset_val = datasets.create(name_val, root, split_id=split_id, mode=args.dataset_mode)
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
        T.RandomCropFlip(height, width, area=args.area),
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
                     transform=train_transformer,
                     has_npy=npy),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentityWeightedSampler(train_set, num_instances, batch_size=batch_size),
        pin_memory=pin_memory, drop_last=True)

    fnames = np.asarray(train_set)[:, 0]
    fname2ind = dict(zip(fnames, np.arange(fnames.shape[0])))
    setattr(train_loader, 'fname2ind', fname2ind)

    val_loader = DataLoader(
        Preprocessor(dataset_val.val, root=dataset_val.images_dir,
                     transform=test_transformer,
                     has_npy=npy),
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
                     transform=test_transformer,
                     has_npy=npy),
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
    writer = SummaryWriter(args.logs_dir)

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'

    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args)
    # Create model

    base_model = models.create(args.arch,
                               dropout=args.dropout,
                               pretrained=args.pretrained,
                               cut_at_pooling=True, bottleneck=args.bottleneck
                               , convop=args.convop
                               )
    if args.branchs * args.branch_dim != 0:
        local_model = Mask(base_model.out_planes, args.branchs, args.branch_dim,
                           height=args.height // 32,
                           width=args.width // 32,
                           dopout=args.dropout
                           )
    else:
        local_model = None
    if args.global_dim != 0:
        global_model = Global(base_model.out_planes, args.global_dim, dropout=args.dropout)
    else:
        global_model = None
    lomo_model = LomoNet() if args.has_npy else None

    # np.random.seed(16)

    def get_controller(scale=args.scale, translation=args.translation, theta=args.theta):
        controller = []
        for sx in scale:
            for sy in scale:
                for tx in translation:
                    for ty in translation:
                        for th in theta:
                            controller.append([sx * np.cos(th), -sx * np.sin(th), tx,
                                               sy * np.sin(th), sy * np.cos(th), ty])
        print('controller stride is ', len(controller))
        controller = np.stack(controller)
        controller = controller.reshape(-1, 2, 3)
        controller = np.ascontiguousarray(controller, np.float32)
        return controller

    dconv_model = ZPC1Conv(2048, args.double, kernel_size=3, vector=True
                           ).cuda() if args.double else None
    # dconv_model = TC1Conv(2048, args.double, kernel_size=3, vector=True
    #                        ).cuda() if args.double else None
    # dconv_model = None
    concat_inplates = args.branchs * args.branch_dim + args.global_dim
    if args.double:
        concat_inplates += args.double
    if args.has_npy:
        concat_inplates += 256
    concat_model = ConcatReduce(concat_inplates,
                                args.num_classes, dropout=0)

    model = SingleNet(base_model, global_model, local_model,
                      lomo_model,
                      dconv_model,
                      concat_model=concat_model)

    print(model)
    param_mb = sum(p.numel() for p in model.parameters()) / 1000000.0
    logging.info('    Total params: %.2fM' % (param_mb))
    writer.add_scalar('param', param_mb, global_step=0)
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
    if args.gpu is None:
        model = nn.DataParallel(model)
    elif len(args.gpu) == 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = nn.DataParallel(model, device_ids=range(len(args.gpu))).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model, gpu=args.gpu)
    if args.evaluate:
        acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric, final=True)
        # acc = evaluator.evaluate(val_loader, dataset.val, dataset.val, metric, final=True)
        lz.logging.info('eval cmc-1 is {}'.format(acc))
        # db.close()
        return 0

    # Criterion
    if args.gpu is not None:
        criterion = TripletLoss(margin=args.margin).cuda()
    else:
        criterion = TripletLoss(margin=args.margin)

    # Optimizer
    #
    # for param in itertools.chain(model.module.base.parameters(),
    #                              model.module.feat.parameters(),
    #                              model.module.feat_bn.parameters(),
    #                              model.module.classifier.parameters()
    #                              ):
    #     param.requires_grad = False

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  # module.stn
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay, momentum=0.9,
                                    nesterov=True)
    else:
        raise NotImplementedError
    # Trainer
    trainer = Trainer(model, criterion, dbg=False, logs_at=args.logs_dir + '/vis', loss_div_weight=args.loss_div_weight)

    # Schedule learning rate
    def adjust_lr(epoch, optimizer=optimizer, base_lr=args.lr, steps=args.steps, decay=args.decay):

        exp = len(steps)
        for i, step in enumerate(steps):
            if epoch < step:
                exp = i
                break
        lr = base_lr * decay ** exp

        lz.logging.info('use lr {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch=epoch)
        if args.hard_examples:
            # Use sequential train set loader
            data_loader = DataLoader(
                Preprocessor(dataset.train, root=dataset.images_dir,
                             transform=val_loader.dataset.transform),
                batch_size=args.batch_size, num_workers=args.workers,
                shuffle=False, pin_memory=False)
            # Mine hard triplet examples, index of [(anchor, pos, neg), ...]
            triplets = mine_hard_triplets(model,
                                          data_loader, margin=args.margin, batch_size=args.batch_size)
            print("Mined {} hard example triplets".format(len(triplets)))
            # Build a hard examples loader
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                num_workers=train_loader.num_workers,
                sampler=SubsetRandomSampler(np.unique(np.asarray(triplets).ravel())),
                pin_memory=True, drop_last=True)

            # RandomIdentityWeightedSampler(train_loader.dataset.dataset,
            #                               args.num_instances,
            #                               batch_size=args.batch_size,
            #                               subsample=triplets),

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

        mAP, acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)
        writer.add_scalar('test/top-1', acc, epoch)
        writer.add_scalar('test/mAP', mAP, epoch)

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
    mAP, acc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)
    writer.add_scalar('test/top-1', acc, args.epochs)
    writer.add_scalar('test/mAP', mAP, args.epochs)
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
