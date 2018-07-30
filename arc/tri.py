import sys

sys.path.insert(0, '/data1/xinglu/prj/open-reid')

from lz import *
import lz
from torch.optim import Optimizer
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
from reid.utils.serialization import *
from reid.utils.dop import DopInfo

from tensorboardX import SummaryWriter


def run(_):
    cfgs = lz.load_cfg('./cfgs/single_ohnm.py')
    procs = []
    for args in cfgs.cfgs:
        if args.loss != 'tcx':
            print(f'skip {args.loss} {args.logs_dir}')
            continue
        # args.log_at = np.concatenate([
        #     args.log_at,
        #     range(args.epochs - 8, args.epochs, 1)
        # ])
        args.logs_dir = 'work/' + args.logs_dir
        if osp.exists(args.logs_dir) and osp.exists(args.logs_dir + '/checkpoint.64.pth'):
            os.listdir(args.logs_dir)
            continue

        if not args.gpu_fix:
            args.gpu = lz.get_dev(n=len(args.gpu),
                                  ok=args.gpu_range,
                                  mem=[0.12, 0.07], sleep=32.3)
        lz.logging.info(f'use gpu {args.gpu}')
        # args.batch_size = 16
        # args.gpu = (3, )
        # args.epochs = 1
        # args.logs_dir+='.bak'

        if isinstance(args.gpu, int):
            args.gpu = [args.gpu]
        if not args.evaluate and not args.vis:
            assert args.logs_dir != args.resume
            lz.mkdir_p(args.logs_dir, delete=True)
            lz.pickle_dump(args, args.logs_dir + '/conf.pkl')
        if not cfgs.parallel:
            main(args)
        else:
            proc = mp.Process(target=main, args=(args,))
            proc.start()
            lz.logging.info('next')
            time.sleep(random.randint(39, 90))
            procs.append(proc)

    for proc in procs:
        proc.join()


def main(args):
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    sys.stderr = Logger(osp.join(args.logs_dir, 'err.txt'))
    lz.init_dev(args.gpu)
    print('config is {}'.format(vars(args)))
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'

    dataset = datasets.create(args.dataset, root=args.root + '/' + args.dataset,
                              split_id=args.split_id, mode=args.dataset_mode)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval
    num_classes = dataset.num_trainval_ids

    train_transformer = T.Compose([
        T.RandomCropFlip(args.height, args.width, area=args.area),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer,
                     ),
        batch_size=args.batch_size, num_workers=args.workers,
        sampler=RandomIdentityWeightedSampler(
            train_set, args.num_instances,
            batch_size=args.batch_size,
            rand_ratio=args.rand_ratio,
        ),
        # shuffle=True,
        pin_memory=args.pin_memory, drop_last=True)

    test_loader = DataLoader(
        Preprocessor(dataset.val,
                     root=dataset.images_dir,
                     transform=test_transformer,
                     ),
        batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=False)

    # Create model
    model = models.create(args.arch,
                          dropout=args.dropout,
                          pretrained=args.pretrained,
                          block_name=args.block_name,
                          block_name2=args.block_name2,
                          num_features=args.num_classes,
                          num_classes=num_classes,
                          num_deform=args.num_deform,
                          fusion=args.fusion,
                          )

    print(model)
    param_mb = sum(p.numel() for p in model.parameters()) / 1000000.0
    logging.info('    Total params: %.2fM' % (param_mb))

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        while not osp.exists(args.resume):
            lz.logging.warning(' no chkpoint {} '.format(args.resume))
            time.sleep(20)
        if torch.cuda.is_available():
            checkpoint = load_checkpoint(args.resume)
        else:
            checkpoint = load_checkpoint(args.resume, map_location='cpu')
        # model.load_state_dict(checkpoint['state_dict'])
        db_name = args.logs_dir + '/' + args.logs_dir.split('/')[-1] + '.h5'
        load_state_dict(model, checkpoint['state_dict'])
        with lz.Database(db_name) as db:
            if 'cent' in checkpoint:
                db['cent'] = to_numpy(checkpoint['cent'])
            db['xent'] = to_numpy(checkpoint['state_dict']['embed2.weight'])
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
    evaluator = Evaluator(model, gpu=args.gpu, conf=args.eval_conf, args=args)
    if args.evaluate:
        res = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric,
                                 final=True, prefix='test')

        lz.logging.info('eval {}'.format(res))
        return res
    # Criterion
    if not args.xent_smooth:
        xent = nn.CrossEntropyLoss()
    else:
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    setattr(xent, 'name', 'xent')

    criterion = [TripletLoss(margin=args.margin, mode='hard', args=args),
                 CenterLoss(num_classes=num_classes, feat_dim=args.num_classes,
                            margin2=args.margin2,
                            margin3=args.margin3, mode=args.mode,
                            push_scale=args.push_scale,
                            args=args),
                 xent
                 ]
    if args.gpu is not None:
        criterion = [c.cuda() for c in criterion]
    # Optimizer
    fast_params = []
    for name, param in model.named_parameters():
        if name == 'module.embed1.weight' or name == 'module.embed2.weight':
            fast_params.append(param)
    fast_params_ids = set(map(fid, fast_params))
    normal_params = [p for p in model.parameters() if fid(p) not in fast_params_ids]
    param_groups = [
        {'params': fast_params, 'lr_mult': args.lr_mult},
        {'params': normal_params, 'lr_mult': 1.},
    ]
    if args.optimizer_cent == 'sgd':
        optimizer_cent = torch.optim.SGD(criterion[1].parameters(), lr=args.lr_cent, )
    else:
        optimizer_cent = torch.optim.Adam(criterion[1].parameters(), lr=args.lr_cent, )
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            # model.parameters(),
            param_groups,
            lr=args.lr,
            betas=args.adam_betas,
            eps=args.adam_eps,  # adam hyperparameter
            weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            # filter(lambda p: p.requires_grad, model.parameters()),
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay, momentum=0.9,
            nesterov=True)
    else:
        raise NotImplementedError

    if args.cls_pretrain:
        args_cp = copy.deepcopy(args)
        args_cp.cls_weight = 1
        args_cp.tri_weight = 0
        trainer = XentTrainer(model, criterion, dbg=False,
                              logs_at=args_cp.logs_dir + '/vis', args=args_cp)
        for epoch in range(start_epoch, args_cp.epochs):
            hist = trainer.train(epoch, train_loader, optimizer)
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'cent': criterion[1].centers,
                'epoch': epoch + 1,
                'best_top1': best_top1,
            }, True, fpath=osp.join(args.logs_dir, 'checkpoint.{}.pth'.format(epoch)))  #
            print('Finished epoch {:3d} hist {}'.
                  format(epoch, hist))
    # Trainer
    trainer = TCXTrainer(model, criterion, dbg=True,
                         logs_at=args.logs_dir + '/vis', args=args, dop_info=dop_info)

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

    def adjust_bs(epoch, args):
        if args.batch_size_l == []:
            return args
        res = 0
        for i, step in enumerate(args.bs_steps):
            if epoch > step:
                res = i + 1
        print(epoch, res)
        if res >= len(args.num_instances_l):
            res = -1
        args.batch_size = args.batch_size_l[res]
        args.num_instances = args.num_instances_l[res]
        return args

    writer = SummaryWriter(args.logs_dir)
    writer.add_scalar('param', param_mb, global_step=0)

    # schedule = CyclicLR(optimizer)
    schedule = None
    # Start training
    for epoch in range(start_epoch, args.epochs):
        # warm up
        # mAP, acc,rank5 = evaluator.evaluate(val_loader, dataset.val, dataset.val, metric)

        adjust_lr(epoch=epoch)
        args = adjust_bs(epoch, args)

        hist = trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, schedule=schedule,
                             optimizer_cent=optimizer_cent)
        for k, v in hist.items():
            writer.add_scalar('train/' + k, v, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('bs', args.batch_size, epoch)
        writer.add_scalar('num_instances', args.num_instances, epoch)

        if not args.log_middle:
            continue
        if epoch < args.start_save:
            continue
        if epoch % 15 == 0:
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'cent': criterion[1].centers,
                'epoch': epoch + 1,
                'best_top1': best_top1,
            }, False, fpath=osp.join(args.logs_dir, 'checkpoint.{}.pth'.format(epoch)))

        if epoch not in args.log_at:
            continue

        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'cent': criterion[1].centers,
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, False, fpath=osp.join(args.logs_dir, 'checkpoint.{}.pth'.format(epoch)))

        # res = evaluator.evaluate(val_loader, dataset.val, dataset.val, metric)
        # for n, v in res.items():
        #     writer.add_scalar('train/'+n, v, epoch)

        res = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric, epoch=epoch)
        for n, v in res.items():
            writer.add_scalar('test/' + n, v, epoch)

        top1 = res['top-1']
        is_best = top1 > best_top1

        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'cent': criterion[1].centers,
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.{}.pth'.format(epoch)))  #

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    res = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)
    for n, v in res.items():
        writer.add_scalar('test/' + n, v, args.epochs)

    if osp.exists(osp.join(args.logs_dir, 'model_best.pth')) and args.test_best:
        print('Test with best model:')
        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth'))
        model.module.load_state_dict(checkpoint['state_dict'])
        metric.train(model, train_loader)
        res = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric, final=True)
        for n, v in res.items():
            writer.add_scalar('test/' + n, v, args.epochs + 1)
        lz.logging.info('final eval is {}'.format(res))

    writer.close()
    json_dump(res, args.logs_dir + '/res.json', 'w')
    return res


if __name__ == '__main__':
    import datetime

    tic = time.time()
    run('')
    toc = time.time()
    print('consume time ', toc - tic)
    if toc - tic > 600:
        mail('tri center xent finish')
    print(datetime.datetime.now().strftime('%D-%H:%M:%S'))
