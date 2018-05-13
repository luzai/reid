import sys

sys.path.insert(0, '/data1/xinglu/prj/open-reid')

from lz import *

cfgs = [
    # edict(
    #     logs_dir='cu03det.cent.eval',
    #     dataset='cu03det', optimizer='adam', lr=3e-4,
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     dropout=0, loss='tri_center', mode='cent',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, weight_dis_cent=0, lr_cent=5e-1, weight_cent=1e-3, gpu_range=range(4),
    #     push_scale=1.,
    #     evaluate=True,
    #     resume='/data2/xinglu/work/reid/work/cu03det.cent/model_best.pth',
    # ),
    #
    # edict(
    #     logs_dir='cu03det.xent.eval',
    #     dataset='cu03det', optimizer='adam', lr=3e-4,
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     dropout=0, loss='tri_xent', xent_smooth=True, gpu_range=range(4),
    #     cls_weight=1, tri_weight=0, lr_mult=10,
    #     evaluate=True,
    #     resume='/data2/xinglu/work/reid/work/cu03det.xent/model_best.pth',
    # ),

    # edict(
    #     logs_dir='tri.viper.vis',
    #     dataset='viper', dataset_val='viper', lr=3e-4, margin=0.5, area=(0.85, 1),
    #     steps=[40, 60], epochs=65,
    #     arch='resnet50',
    #     # log_at=[64, 65, 66],
    #     log_at=np.arange(0, 66, 9),
    #     batch_size=128, num_instances=2, gpu=range(1), num_classes=128,
    #     dropout=0, loss='tri_center', mode='ccent.min',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, weight_dis_cent=0, lr_cent=0, weight_cent=0, gpu_range=range(4),
    #     push_scale=1.,
    #     # evaluate=True,
    # )
]

# cfg = edict(
#     logs_dir='cfisher',
#     dataset='cu03lbl',
#     log_at=[0, 30, 64, 65, 66],
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     dropout=0, loss='tri_center', mode='ccent',
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, weight_dis_cent=0, lr_cent=5e-1, weight_cent=.5, gpu_range=range(3),
#     push_scale=1.,
# )
# for dataset, weight_cent, dop, dis, scale, mode in grid_iter(
#         ['cu03lbl', 'cu03det', ],
#         [1e-1, 0],
#         [1, ],
#         [0, ],
#         [1e-2, ],
#         ['ccent.min'],
# ):
#     cfg_t = copy.deepcopy(cfg)
#     cfg_t.weight_cent = weight_cent
#     cfg_t.random_ratio = dop
#     cfg_t.dataset = dataset
#     cfg_t.weight_dis_cent = dis
#     cfg_t.push_scale = scale
#     cfg_t.mode = mode
#     cfg_t.logs_dir = f'{cfg.logs_dir}.{dataset}.{weight_cent:.0e}.dop{dop:.1f}.dis.all{dis:.0e}.{mode}.{scale:.0e}'
#     cfgs.append(cfg_t)

# cfg = edict(
#     logs_dir='cfisher.scale',
#     dataset='viper',
#     log_at=[0, 30, 64, 65, 66],
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     dropout=0, loss='tri_center', mode='ccent',
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, weight_dis_cent=0, lr_cent=5e-1, weight_cent=.5, gpu_range=range(4),
#     push_scale=1.,
# )
# for dataset, weight_cent, dop, dis, scale, mode in grid_iter(
#         # ['viper', 'cuhk01'],
#         ['cu03det'],
#         [1e-1, ],
#         [1, ],
#         [0, ],
#         [1e-1, 1e-3],
#         ['ccent.min', 'ccent.all'],
# ):
#     cfg_t = copy.deepcopy(cfg)
#     cfg_t.weight_cent = weight_cent
#     cfg_t.random_ratio = dop
#     cfg_t.dataset = dataset
#     cfg_t.weight_dis_cent = dis
#     cfg_t.push_scale = scale
#     cfg_t.mode = mode
#     cfg_t.logs_dir = f'{cfg.logs_dir}.{dataset}.{weight_cent:.0e}.{scale:.0e}.{mode}'  # .dop{dop:.1f}.dis.all{dis:.0e}
#     cfgs.append(cfg_t)

# cfg = edict(
#     logs_dir='multis',
#     dataset='cu03det',
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     dropout=0.5, loss='tri_center',
#     cls_weight=0, tri_weight=1, lr_cent=0, weight_cent=0, gpu_range=(3,),
#     random_ratio=1, weight_dis_cent=0,
#     embed="concat",
#     block_name='SEBottleneck', block_name2='SEBottleneck',
#     log_at=[0, 30, 64, 65, 66],
# )
#
# for dataset, embed, bn in grid_iter(
#         ['viper', 'cuhk01', 'cu03det', 'cu03lbl', 'market1501'],
#         [None, 'concat'],
#         ['SEBottleneck', 'Bottleneck'],
# ):
#     cfg_t = copy.deepcopy(cfg)
#     if embed == 'concat' and bn == 'Bottleneck':
#         continue
#     cfg_t.dataset = dataset
#     cfg_t.embed = embed
#     cfg_t.block_name = bn
#     cfg_t.block_name2 = bn
#
#     cfg_t.logs_dir = f'{cfg.logs_dir}.{dataset}.{bn}.{embed}'
#     cfgs.append(cfg_t)

base = edict(
    weight_lda=None, test_best=True, push_scale=1.,
    lr=3e-4, margin=0.5, area=(0.85, 1), margin2=0.4, margin3=1.3,
    steps=[40, 60], epochs=65,
    arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck', convop='nn.Conv2d',
    weight_dis_cent=0,
    weight_cent=0, lr_cent=0.5, xent_smooth=False,

    lr_mult=1., fusion=None, eval_conf='cuhk03',
    cls_weight=0., random_ratio=1, tri_weight=1, num_deform=3, cls_pretrain=False,
    bs_steps=[], batch_size_l=[], num_instances_l=[],
    scale=(1,), translation=(0,), theta=(0,),
    hard_examples=False, has_npy=False, double=0, loss_div_weight=0,
    pretrained=True, dbg=False, data_dir='/home/xinglu/.torch/data',
    restart=True, workers=8, split=0, height=256, width=128,
    combine_trainval=True, num_instances=4,
    evaluate=False, dropout=0,
    # log_at=np.concatenate([
    #     range(0, 640, 21),
    # ]),
    log_at=[],
    weight_decay=5e-4, resume=None, start_save=0,
    seed=None, print_freq=3, dist_metric='euclidean',
    branchs=0, branch_dim=64, global_dim=1024, num_classes=128,
    loss='tri',
    # mode='hard',
    mode='ccent',
    gpu=(0,), pin_mem=True, log_start=False, log_middle=True, gpu_range=range(4),
    # tuning
    dataset='market1501', dataset_mode='combine', dataset_val='market1501',
    batch_size=128, logs_dir='', embed="concat",
    optimizer='adam', normalize=True, decay=0.1,
)

for k, v in enumerate(cfgs):
    v = dict_update(base, v)
    cfgs[k] = edict(v)

for args in cfgs:
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
    elif args.dataset == 'mkt':
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
    elif args.dataset == 'cuhk01':
        args.dataset = 'cuhk01'
        args.dataset_val = 'cuhk01'
    elif args.dataset == 'dukemtmc':
        args.dataset = 'dukemtmc'
        args.dataset_val = 'dukemtmc'


def format_cfg(cfg):
    if cfg.gpu is not None:
        cfg.pin_mem = True
        cfg.workers = len(cfg.gpu) * 8
    else:
        cfg.pin_mem = False
        cfg.workers = 4


def is_all_same(lst):
    res = [lsti == lst[0] for lsti in lst]
    return np.asarray(res).all()


if __name__ == '__main__':
    import tabulate

    df = pd.DataFrame(cfgs)
    if len(cfgs) == 1:
        print(df)
        exit(0)
    res = []
    for j in range(df.shape[1]):
        if not is_all_same(df.iloc[:, j].tolist()): res.append(j)
    res = [df.columns[r] for r in res]
    df1 = df[res]
    df1.index = df1.logs_dir
    del df1['logs_dir']
    print(tabulate.tabulate(df1, headers="keys", tablefmt="pipe"))
