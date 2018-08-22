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
    cfgs = lz.load_cfg('./cfgs/football.py')
    procs = []
    for args in cfgs.cfgs:
        if args.loss != 'tcx' and args.loss != 'tri':
            print(f'skip {args.loss} {args.logs_dir}')
            continue

        args.log_at = np.concatenate([
            args.log_at,
            range(args.epochs - 8, args.epochs, 1)
        ])
        args.logs_dir = lz.work_path + 'reid/work/' + args.logs_dir
        if osp.exists(args.logs_dir) and osp.exists(args.logs_dir + '/checkpoint.64.pth'):
            print(os.listdir(args.logs_dir))
            continue

        if not args.gpu_fix:
            args.gpu = lz.get_dev(n=len(args.gpu),
                                  ok=args.gpu_range,
                                  mem_thresh=[0.12, 0.20], sleep=32.3)
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
        if cfgs.no_proc:
            main(args)
        else:
            proc = mp.Process(target=main, args=(args,))
            proc.start()
            lz.logging.info('next')
            time.sleep(random.randint(39, 90))
            if not cfgs.parallel:
                proc.join()
            else:
                procs.append(proc)

    if cfgs.parallel:
        for proc in procs:
            proc.join()


def get_data(args):
    (name, split_id,
     data_dir, height, width,
     batch_size, num_instances,
     workers, combine_trainval) = (
        args.dataset, args.split,
        args.data_dir, args.height, args.width,
        args.batch_size, args.num_instances,
        args.workers, args.combine_trainval,)
    pin_memory = args.pin_mem
    name_val = args.dataset_val or args.dataset
    npy = args.has_npy
    rand_ratio = args.random_ratio
    if isinstance(name, list):
        dataset = datasets.creates(name, split_id=split_id)
    else:
        dataset = datasets.create(name, split_id=split_id, )

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])
    test_loader = DataLoader(
        Preprocessor(dataset.query,
                     transform=test_transformer,
                     has_npy=npy),
        batch_size=batch_size,  # * 2
        num_workers=workers,
        shuffle=False, pin_memory=False)
    return test_loader


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

    test_loader = get_data(args)

    # Create model
    model = models.create(args.arch,
                          dropout=args.dropout,
                          pretrained=args.pretrained,
                          block_name=args.block_name,
                          block_name2=args.block_name2,
                          num_features=args.num_classes,
                          num_classes=100,
                          num_deform=args.num_deform,
                          fusion=args.fusion,
                          last_conv_stride=args.last_conv_stride,
                          last_conv_dilation=args.last_conv_dilation,
                          )

    print(model)
    param_mb = sum(p.numel() for p in model.parameters()) / 1000000.0
    print('    Total params: %.2fM' % (param_mb))

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
    if args.gpu is None or len(args.gpu) == 0:
        model = nn.DataParallel(model)
    elif len(args.gpu) == 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = nn.DataParallel(model, device_ids=range(len(args.gpu))).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    features, _ = extract_features(model, test_loader)
    for k in features.keys():
        features[k] = features[k].numpy()
    lz.msgpack_dump(features, work_path + '/reid.person/fea.mp', allow_np=True)
