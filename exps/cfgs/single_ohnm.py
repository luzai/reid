# cython: boundscheck=False, wraparound=False, nonecheck=False

import sys

sys.path.insert(0, '/data1/xinglu/prj/open-reid')

from lz import *

parallel = True

cfgs = [

    # edict(
    #     logs_dir='eval3',
    #     dataset='mkt', log_at=[0, 1, 2, 30, 64, 65],
    #     epochs=65, steps=[20, 40],
    #     batch_size=128, num_instances=4, gpu=(2,), num_classes=128,
    #     dropout=0, loss='tcx', mode='ccent.ccentall.disall',
    #     cls_weight=0, tri_weight=0, lr_mult=10., xent_smooth=True,
    #     random_ratio=.5,
    #     weight_dis_cent=0, weight_cent=1,
    #     gpu_range=range(4), gpu_fix=False,
    #     push_scale=1, embed=None, margin2=0.05,
    #     lr=1e-2, optimizer='sgd',
    #     lr_cent=1e-2, optimizer_cent='sgd',
    #     # lr=3e-4, optimizer='adam',
    #     # lr_cent=3e-4, optimizer_cent='adam',
    #     evaluate=True, vis=False,
    #     # resume='work/final.xent.mkt.smthTrue/model_best.pth'
    #     resume='work/tri.margin.mkt.mg0.5.mg2_1.0.mg3_1.0/model_best.pth'
    #     # resume='work/final.tri.mkt/model_best.pth'
    # ),

    edict(
        logs_dir='tri.advtrue',
        dataset='cu03det',
        log_at=[0, 30, 64, 65, 66], gpu_fix=True,
        batch_size=128, num_instances=4, gpu=(0,1), num_classes=128,
        dropout=0, loss='tcx', mode='',
        cls_weight=0, tri_weight=1, weight_dis_cent=0, weight_cent=0,
        random_ratio=1, lr_cent=0,
        gpu_range=range(4), lr_mult=1,
        push_scale=1., embed=None, margin=.5, margin2=1., margin3=1.,
    )

]

# cfg = edict(
#     logs_dir='tri.margin.6',
#     dataset='cu03det',
#     log_at=[0, 30, 64, 65, 66],
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     dropout=0, loss='tcx', mode='',
#     cls_weight=0, tri_weight=1, weight_dis_cent=0, weight_cent=0,
#     random_ratio=1, lr_cent=0,
#     gpu_range=range(4), lr_mult=1,
#     push_scale=1., embed=None, margin=.5, margin2=1., margin3=1.,
# )
#
# for (dataset,
#      margin,
#      margin2,
#      margin3
#      ) in grid_iter(
#     ['cu03det'],
#     [0., ],
#     [1.1, ],
#     [1.1, ]
# ):
#     cfg_t = copy.deepcopy(cfg)
#     cfg_t.dataset = dataset
#     cfg_t.margin = margin
#     cfg_t.margin2 = margin2
#     cfg_t.margin3 = margin3
#     cfg_t.logs_dir = f'{cfg.logs_dir}.{dataset}.mg{margin}.mg2_{margin2}.mg3_{margin3}'
#     cfgs.append(cfg_t)
#
# for (dataset,
#      margin,
#      margin2,
#      margin3
#      ) in grid_iter(
#     ['cu03det'],
#     [.5, ],
#     [1., ],
#     [1., ]
# ):
#     cfg_t = copy.deepcopy(cfg)
#     cfg_t.dataset = dataset
#     cfg_t.margin = margin
#     cfg_t.margin2 = margin2
#     cfg_t.margin3 = margin3
#     cfg_t.logs_dir = f'{cfg.logs_dir}.{dataset}.mg{margin}.mg2_{margin2}.mg3_{margin3}'
#     cfgs.append(cfg_t)
#
# for (dataset,
#      margin,
#      margin2,
#      margin3,
#      lr_mult,
#      ) in grid_iter(
#     ['mkt', ],
#     [.0, .3, .5],
#     [.9, 1.2, 1.05, 1.1],
#     [.9, 1.2, 1.05, 1.1],
#     [1, 2, 4],
# ):
#     cfg_t = copy.deepcopy(cfg)
#     cfg_t.dataset = dataset
#     cfg_t.margin = margin
#     cfg_t.margin2 = margin2
#     cfg_t.margin3 = margin3
#     cfg_t.lr_mult = lr_mult
#     cfg_t.logs_dir = f'{cfg.logs_dir}.{dataset}.mg{margin}.mg2_{margin2}.mg3_{margin3}.lrm_{lr_mult}'
#     cfgs.append(cfg_t)

# cfg = edict(
#     logs_dir='final.xent',
#     dataset='dukemtmc',
#     log_at=[0, 30, 64, 65, 66],
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     dropout=0, loss='tri_xent', mode='ccent.all.all',
#     cls_weight=1, tri_weight=0,
#     random_ratio=1, weight_dis_cent=0, lr_cent=0, weight_cent=0, gpu_range=(0, 1,),
#     push_scale=1., embed=None
# )
# # cfgs.append(cfg)
# for (dataset,
#      weight_cent,
#      dop,
#      dis,
#      scale,
#      mode, smooth) in grid_iter(
#     # ['cu01easy', 'cu01hard'],
#     ['cu03lbl', 'cu03det', 'mkt', 'dukemtmc'],
#     [0, ],  # weight_cent
#     [1, ],  # dop
#     [0, ],  # dis
#     [1, ],  # scale
#     ['ccent.all.all'],  # cent_mode
#     [True, False]
# ):
#     cfg_t = copy.deepcopy(cfg)
#     cfg_t.weight_cent = weight_cent
#     cfg_t.random_ratio = dop
#     cfg_t.dataset = dataset
#     cfg_t.weight_dis_cent = dis
#     cfg_t.push_scale = scale
#     cfg_t.mode = mode
#     cfg_t.xent_smooth = smooth
#     cfg_t.logs_dir = f'{cfg.logs_dir}.{dataset}.smth{smooth}'
#     cfgs.append(cfg_t)

base = edict(
    optimizer_cent='adam', topk=5, test_best=True,
    push_scale=1., gpu_fix=False, test_batch_size=8,
    lr=3e-4, margin=0.5, area=(0.85, 1),
    margin2=1., margin3=1.,
    steps=[40, 60], epochs=65,
    arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck', convop='nn.Conv2d',
    weight_dis_cent=0, vis=False,
    weight_cent=0, lr_cent=0.5, xent_smooth=False,
    adam_betas=(.9, .999), adam_eps=1e-8,
    lr_mult=1., fusion=None, eval_conf='market1501',
    cls_weight=0., random_ratio=1, tri_weight=1, num_deform=3, cls_pretrain=False,
    bs_steps=[], batch_size_l=[], num_instances_l=[],
    scale=(1,), translation=(0,), theta=(0,),
    hard_examples=False, has_npy=False, double=0, loss_div_weight=0,
    pretrained=True, dbg=False, data_dir='/home/xinglu/.torch/data',
    restart=True, workers=8, split=0, height=256, width=128,
    combine_trainval=True, num_instances=4,
    evaluate=False, dropout=0,
    seq_len=15, vid_pool='avg',
    # log_at=np.concatenate([
    #     range(0, 640, 21),
    # ]),
    log_at=[],
    weight_decay=5e-4, resume=None, start_save=0,
    seed=None, print_freq=3, dist_metric='euclidean',
    branchs=0, branch_dim=64, global_dim=1024, num_classes=128,
    loss='tri',
    # tri_mode = 'hard', cent_mode = 'ccent.all.all',
    mode='ccent.all.all',
    gpu=(0,), pin_mem=True, log_start=False, log_middle=True, gpu_range=range(4),
    # tuning
    dataset='market1501', dataset_mode=None, dataset_val=None,
    batch_size=128, logs_dir='', embed=None,
    optimizer='adam', normalize=True, decay=0.1,
)

for ind, v in enumerate(cfgs):
    v = dict_update(base, v)
    cfgs[ind] = edict(v)

for ind, args in enumerate(cfgs):
    if args.dataset == 'cu03det':
        args.dataset = 'cuhk03'
        args.dataset_val = 'cuhk03'
        args.dataset_mode = 'detect'
        args.eval_conf = 'cuhk03'
    elif args.dataset == 'cu03lbl':
        args.dataset = 'cuhk03'
        args.dataset_val = 'cuhk03'
        args.dataset_mode = 'label'
        args.eval_conf = 'cuhk03'
    elif args.dataset == 'mkt' or args.dataset == 'market' or args.dataset == 'market1501':
        args.dataset = 'market1501'
        args.dataset_val = 'market1501'
        args.eval_conf = 'market1501'
    elif args.dataset == 'msmt':
        args.dataset = 'msmt17'
        args.dataset_val = 'market1501'
        args.eval_conf = 'market1501'
    elif args.dataset == 'cdm':
        args.dataset = 'cdm'
        args.dataset_val = 'market1501'
        args.eval_conf = 'market1501'
    elif args.dataset == 'viper':
        args.dataset = 'viper'
        args.dataset_val = 'viper'
        args.eval_conf = 'market1501'
    elif args.dataset == 'cu01hard':
        args.dataset = 'cuhk01'
        args.dataset_val = 'cuhk01'
        args.eval_conf = 'cuhk03'
        args.dataset_mode = 'hard'
    elif args.dataset == 'cu01easy':
        args.dataset = 'cuhk01'
        args.dataset_val = 'cuhk01'
        args.eval_conf = 'cuhk03'
        args.dataset_mode = 'easy'
    elif args.dataset == 'dukemtmc':
        args.dataset = 'dukemtmc'
        args.dataset_val = 'dukemtmc'
        args.eval_conf = 'market1501'
    else:
        # raise ValueError(f'dataset ... {args.dataset}')
        args.dataset_val = args.dataset
        args.eval_conf = 'market1501'
    cfgs[ind] = edict(args)

for ind, args in enumerate(cfgs):
    if args.evaluate is True and args.resume is not None and osp.exists(args.resume + '/conf.pkl'):
        resume = args.resume
        # logs_dir = args.logs_dir
        gpu_range = args.gpu_range
        gpu_fix = args.gpu_fix
        gpu = args.gpu
        bs = args.batch_size
        vis = args.vis
        args_ori = pickle_load(args.resume + '/conf.pkl')
        args = dict_update(args, args_ori)
        args = edict(args)
        args.resume = resume + '/model_best.pth'
        args.evaluate = True
        args.logs_dir = args.logs_dir.replace('work/', '') + '/eval'
        args.gpu_range = gpu_range
        args.gpu_fix = gpu_fix
        args.gpu = gpu
        args.batch_size = bs
        args.vis = vis
        cfgs[ind] = args


# def format_cfg(cfg):
#     if cfg.gpu is not None:
#         cfg.pin_mem = True
#         cfg.workers = len(cfg.gpu) * 8
#     else:
#         cfg.pin_mem = False
#         cfg.workers = 4


def is_all_same(lst):
    res = [lsti == lst[0] for lsti in lst]
    try:
        return np.asarray(res).all()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    import tabulate

    df = pd.DataFrame(cfgs)
    if len(cfgs) == 1:
        print(df)
        exit(0)
    res = []
    for j in range(df.shape[1]):
        if not is_all_same(df.iloc[:, j].tolist()):
            res.append(j)
    res = [df.columns[r] for r in res]
    df1 = df[res]
    df1.index = df1.logs_dir
    # del df1['logs_dir']

    print(tabulate.tabulate(df1.sort_values(by='logs_dir'), headers="keys", tablefmt="pipe"))
