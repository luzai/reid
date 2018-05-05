import sys

sys.path.insert(0, '/data1/xinglu/prj/open-reid')

from lz import *

cfgs = [

    edict(
        logs_dir='cu03det.bak',
        dataset='cu03det',
        batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
        dropout=0, loss='tri_center', mode='ccent',
        cls_weight=0, tri_weight=1,
        random_ratio=1, weight_dis_cent=0, lr_cent=5e-1, weight_cent=1e-3, gpu_range=range(4),
        # evaluate=True,
        # resume = '/data1/xinglu/prj/open-reid/exps/work/cu03det.mglmean.0e+00.1e-01.run0/model_best.pth',
        # resume = '/data1/xinglu/prj/open-reid/exps/work/cu03det.cent.final.1e-2/model_best.pth',
    ),

]

# cfg = edict(
#     logs_dir='contrast.dis.all',
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     dropout=0, loss='tri_center', mode='ccent',
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, weight_dis_cent=0, lr_cent=5e-1, weight_cent=.5, gpu_range=range(4),
# )
#
# for dataset, weight_cent, dop, dis in grid_iter(
#         ['cu03det', 'cu03lbl', 'mkt'],
#         [1e-1, ],
#         [1],
#         [1e-3, 1e-5],
# ):
#     cfg_t = copy.deepcopy(cfg)
#     cfg_t.weight_cent = weight_cent
#     cfg_t.random_ratio = dop
#     cfg_t.dataset = dataset
#     cfg_t.weight_dis_cent = dis
#
#     cfg_t.logs_dir = f'{cfg.logs_dir}.{dataset}.{weight_cent:.0e}.dop{dop:.1f}.dis{dis:.0e}'
#     cfgs.append(cfg_t)


base = edict(
    weight_lda=None, test_best=True,
    lr=3e-4, margin=0.5, area=(0.85, 1), margin2=0.4, margin3=1.3,
    steps=[40, 60], epochs=65,
    arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck', convop='nn.Conv2d',
    weight_dis_cent=0,
    weight_cent=0, lr_cent=0.5, xent_smooth=False,

    lr_mult=0.1, fusion=None, eval_conf='cuhk03',
    cls_weight=0., random_ratio=1, tri_weight=1, num_deform=3, cls_pretrain=False,
    bs_steps=[], batch_size_l=[], num_instances_l=[],
    scale=(1,), translation=(0,), theta=(0,),
    hard_examples=False, has_npy=False, double=0, loss_div_weight=0,
    pretrained=True, dbg=False, data_dir='/home/xinglu/.torch/data',
    restart=True, workers=8, split=0, height=256, width=128,
    combine_trainval=True, num_instances=4,
    evaluate=False, dropout=0,
    log_at=np.concatenate([
        range(0, 640, 21),
    ]),
    weight_decay=5e-4, resume=None, start_save=0,
    seed=None, print_freq=3, dist_metric='euclidean',
    branchs=0, branch_dim=64, global_dim=1024, num_classes=128,
    loss='tri', mode='hard',
    gpu=(0,), pin_mem=True, log_start=False, log_middle=True, gpu_range=range(4),
    # tuning
    dataset='market1501', dataset_mode='combine', dataset_val='market1501',
    batch_size=128, logs_dir='', embed="concat",
    optimizer='adam', normalize=True, decay=0.1,
)

for k, v in enumerate(cfgs):
    v = dict_update(base, v)
    cfgs[k] = edict(v)


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


import tabulate

if __name__ == '__main__':
    df = pd.DataFrame(cfgs)
    res = []
    for j in range(df.shape[1]):
        if not is_all_same(df.iloc[:, j].tolist()): res.append(j)
    res = [df.columns[r] for r in res]
    df1 = df[res]
    df1.index = df1.logs_dir
    del df1['logs_dir']
    print(tabulate.tabulate(df1, headers="keys", tablefmt="pipe"))
