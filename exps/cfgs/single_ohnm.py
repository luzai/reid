import sys

sys.path.insert(0, '/data1/xinglu/prj/luzai-tool')
sys.path.insert(0, '/data1/xinglu/prj/open-reid')

from lz import *

cfgs = [
    # edict(
    #     logs_dir='msmt17.res.long',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='msmt17', dataset_val='msmt17', eval_conf='market1501',
    #     # dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',dataset_mode = 'label',
    #     # dataset='market1501', dataset_val='market1501', eval_conf='market1501',
    #     # dataset='mars', dataset_val='mars', eval_conf='market1501',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=(0,), num_classes=128,
    #     steps=[80, 120], epochs=125,
    #     workers=4,
    #     dropout=0, loss='tri',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, fusion=None,
    #     log_at=[124, 125],
    #     evaluate=True,
    #     resume='/data1/xinglu/prj/open-reid/exps/work/msmt17.res.2/model_best.pth'
    # ),

    edict(
        logs_dir='market1501.center.vis.2e3.5e-3',
        arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
        dataset='market1501', dataset_val='market1501', eval_conf='market1501',
        lr=3e-4, margin=0.5, area=(0.85, 1), margin2=0.4, margin3=1.3,
        batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
        steps=[40, 60], epochs=65,
        workers=4,
        dataset_mode='label',
        dropout=0, loss='tri_center',
        cls_weight=0, tri_weight=1,
        random_ratio=1, weight_dis_cent=0, lr_cent=2e3, weight_cent=5e-3, gpu_range=range(4),
    ),

    edict(
        logs_dir='market1501.center.vis.5e-1.5e-4',
        arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
        dataset='market1501', dataset_val='market1501', eval_conf='market1501',
        lr=3e-4, margin=0.5, area=(0.85, 1), margin2=0.4, margin3=1.3,
        batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
        steps=[40, 60], epochs=65,
        workers=4,
        dataset_mode='label',
        dropout=0, loss='tri_center',
        cls_weight=0, tri_weight=1,
        random_ratio=1, weight_dis_cent=0, lr_cent=5e-4, weight_cent=5e-4, gpu_range=range(4),
    ),

    # edict(
    #     logs_dir='market1501.discenter.vis',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='market1501', dataset_val='market1501', eval_conf='market1501',
    #     lr=3e-4, margin=0.5, area=(0.85, 1), margin2=0.4, margin3=1.3,
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=4,
    #     dataset_mode='label',
    #     dropout=0, loss='tri_center',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, weight_dis_cent=5e-3, lr_cent=1e3, weight_cent=5e-3, gpu_range=range(4),
    # ),
    #
    # edict(
    #     logs_dir='market1501.concat.dp',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='market1501', dataset_val='market1501', eval_conf='market1501',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=4, dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     loss='tri', weight_cent=0, lr_cent=0.5, weight_dis_cent=0,
    #     random_ratio=1, fusion='concat', gpu_range=(2, 3,)
    # ),

    # edict(
    #     logs_dir='cuhk03detect.res',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=4,
    #     dataset_mode='detect',
    #     dropout=0, loss='tri',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1,
    # ),
    # edict(
    #     logs_dir='cuhk03detect.maskconcat.dp.4.center',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=4,
    #     dataset_mode='detect',
    #     dropout=.4, loss='tri_center',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, weight_dis_cent=0, lr_cent=1e2, weight_cent=5e-2,
    #     fusion='maskconcat',
    # ),
    # edict(
    #     logs_dir='cuhk03detect.maskconcat.dp.4',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=4,
    #     dataset_mode='detect',
    #     dropout=.4, loss='tri',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1,
    #     fusion='maskconcat',
    # ),
    # edict(
    #     logs_dir='cuhk03label.res',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=4,
    #     dataset_mode='label',
    #     dropout=0, loss='tri',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1,
    #     # fusion='maskconcat',
    # ),
    # edict(
    #     logs_dir='cuhk03label.center.tobe',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=4,
    #     dataset_mode='label',
    #     dropout=0, loss='tri_center',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1 / 3., fusion=None, lr_cent=1e2, weight_cent=5e-2
    # ),

    # edict(
    #     logs_dir='cuhk03label.center.mining.5',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=4,
    #     dataset_mode='label',
    #     dropout=0, loss='tri_center',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1 / 2., fusion=None, lr_cent=1e2, weight_cent=5e-2
    # ),

    # edict(
    #     logs_dir='cuhk03label.xent.smooth.adam.128',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=4,
    #     dataset_mode='label', xent_smooth=True,
    #     dropout=0, loss='xent',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, fusion=None,
    # ),

    # edict(
    #     logs_dir='cuhk03label.res.quin',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=4,
    #     dataset_mode='label',
    #     dropout=0, loss='quin',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, fusion=None,
    # ),

    # edict(
    #     logs_dir='market1501.xent.smooth.32.adam',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='market1501', dataset_val='market1501', eval_conf='market1501',
    #     margin=0.5, area=(0.85, 1),
    #     batch_size=32, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=4, dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     loss='xent',
    #     # optimizer='sgd', lr=1e-2,
    #     optimizer='adam', lr=3e-4,
    #     random_ratio=1, fusion=None, xent_smooth=True,
    # ),

    #
    # edict(
    #     logs_dir='market1501.xent.smooth',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='market1501', dataset_val='market1501', eval_conf='market1501',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=4, dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     loss='xent',
    #     random_ratio=1, fusion=None, xent_smooth=True,
    #     # evaluate=True, resume='/data1/xinglu/prj/open-reid/exps/work.3.8/market1501.res/model_best.pth'
    # ),
]

# cfgs_true = []
#
# cfg = cfgs[0]
# for weight_cent, lr_cent in grid_iter([5e-2, 5e-3, 5e-4, 5e-5, 0],
#                                       [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]):
#     print(weight_cent, lr_cent)
#     cfg_t = copy.deepcopy(cfg)
#     cfg_t.weight_cent = weight_cent
#     cfg_t.lr_cent = lr_cent
#     cfg_t.logs_dir = f'{cfg.logs_dir}.{weight_cent:.0e}.{lr_cent:.0e}'
#     cfgs_true.append(cfg_t)
# cfg = cfgs[1]
# for dropout in [.3, .4, .5]:
#     cfg_t = copy.deepcopy(cfg)
#     cfg_t.dropout = dropout
#     cfg_t.logs_dir = f'{cfg_t.logs_dir}.{dropout:.1f}'
#     cfgs_true.append(cfg_t)
# cfgs = cfgs_true

base = edict(
    weight_dis_cent=0,
    weight_cent=0, lr_cent=0.5, xent_smooth=False,
    margin2=0.4, margin3=1.3, margin=0.45,
    lr_mult=0.1, fusion=None, eval_conf='cuhk03',
    cls_weight=0., random_ratio=1, tri_weight=1, num_deform=3, cls_pretrain=False,
    bs_steps=[], batch_size_l=[], num_instances_l=[],
    block_name='Bottleneck', block_name2='Bottleneck', convop='nn.Conv2d',
    scale=(1,), translation=(0,), theta=(0,),
    hard_examples=False, has_npy=False, double=0, loss_div_weight=0,
    pretrained=True, dbg=False, data_dir='/home/xinglu/.torch/data',
    restart=True, workers=4, split=0, height=256, width=128,
    combine_trainval=True, num_instances=4,
    # model
    evaluate=False, dropout=0,
    # optimizer
    lr=3e-4, steps=[100, 150, 160], epochs=65,
    log_at=np.concatenate([
        range(0, 640, 21),
    ]),
    weight_decay=5e-4, resume=None, start_save=0,
    seed=None, print_freq=3, dist_metric='euclidean',
    branchs=0, branch_dim=64, global_dim=1024, num_classes=128,
    loss='tri', mode='hard',
    gpu=(0,), pin_mem=True, log_start=False, log_middle=True, gpu_range=range(4),
    # tuning
    dataset='market1501', dataset_mode='combine', area=(0.85, 1), dataset_val='market1501',
    batch_size=128, logs_dir='', arch='resnet50', embed="concat",
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
