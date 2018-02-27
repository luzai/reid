import sys

sys.path.insert(0, '/data1/xinglu/prj/luzai-tool')
sys.path.insert(0, '/data1/xinglu/prj/open-reid')

from lz import *

cfgs = [


    # edict(
    #     logs_dir='res.basic.scratch',
    #     arch='resnet50', block_name='BasicBlock', block_name2='BasicBlock',
    #     dataset='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8, dataset_mode='combine', dropout=0,
    #     cls_weight=0, tri_weight=1, random_ratio=1,
    #     cls_pretrain=True,
    # ),

    # edict(
    #     logs_dir='rir',
    #     arch='resnet50', block_name='RIRBasicBlock', block_name2='RIRBasicBlock',
    #     dataset='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8, dataset_mode='combine', dropout=0,
    #     cls_weight=0, tri_weight=1, random_ratio=1,
    #     cls_pretrain=True,
    # ),
    #
    # edict(
    #     logs_dir='serir',
    #     arch='resnet50', block_name='SERIRBasicBlock', block_name2='SERIRBasicBlock', dataset='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8, dataset_mode='combine', dropout=0,
    #     cls_weight=0, tri_weight=1, random_ratio=1,
    #     cls_pretrain=True,
    # ),

    # edict(
    #     logs_dir='att_res',
    #     arch='resnet50', block_name='AttResBottleneck', block_name2='AttResBottleneck', dataset='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='combine',
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, cls_pretrain=True,
    # ),

    edict(
        logs_dir='senet.init_senet.dop',
        arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck', dataset='cuhk03',
        lr=3e-4, margin=0.5, area=(0.85, 1),
        dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
        steps=[40, 60], epochs=65,
        workers=0,
        dataset_mode='combine',
        dropout=0,
        cls_weight=0, tri_weight=1,
        random_ratio=0.5,
    ),

    # edict(
    #     logs_dir='resnet.bak',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck', dataset='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='combine',
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1,
    # ),

    # edict(
    #     logs_dir='deform.3.init_zero.lr_mult.0.2',
    #     arch='resnet50', block_name='Bottleneck', block_name2='DeformBottleneck', dataset='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='combine',
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1,
    #     num_deform=3
    # ),


    # edict(
    #     logs_dir='sedeform.all_se.afterlast1x1',
    #     arch='resnet50', block_name='SEBottleneck', block_name2='SEDeformBottleneck', dataset='cuhk03',
    #     lr=3e-4, margi1n=0.5, area=(0.85, 1),
    #     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='combine',
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, num_deform=3,
    # ),

    # edict(
    #     logs_dir='sedeform.3.again.2',
    #     arch='resnet50', block_name='Bottleneck', block_name2='SEDeformBottleneck', dataset='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='combine',
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, num_deform=3,
    # ),

]

# cfgs = [cfgs[-1]]

base = edict(
    cls_weight=0., random_ratio=1, tri_weight=1, num_deform=3,
    cls_pretrain=False,
    bs_steps=[], batch_size_l=[], num_instances_l=[],
    block_name='Bottleneck',
    block_name2='Bottleneck',
    convop='nn.Conv2d',
    scale=(1,), translation=(0,), theta=(0,),
    hard_examples=False,
    has_npy=False,
    double=0, loss_div_weight=0,
    pretrained=True,
    dbg=False,
    data_dir='/home/xinglu/.torch/data',
    restart=True,
    workers=8, split=0, height=256, width=128, combine_trainval=True,
    num_instances=4,
    # model
    evaluate=False,
    dropout=0,
    # loss
    margin=0.45,
    # optimizer
    lr=3e-4,
    steps=[100, 150, 160],
    epochs=165,
    log_at=np.concatenate([
        range(0, 640, 7),
    ]),

    weight_decay=5e-4,
    resume=None,
    start_save=0,
    seed=1, print_freq=3, dist_metric='euclidean',

    branchs=0,
    branch_dim=64,
    global_dim=1024,
    num_classes=128,

    loss='triplet',
    mode='hard',
    gpu=[0, ],
    pin_mem=True,
    log_start=False,
    log_middle=True,
    # tuning
    dataset='market1501',
    dataset_mode='combine',
    area=(0.85, 1),
    dataset_val='market1501',
    batch_size=128,
    logs_dir='',
    arch='resnet50',
    embed="concat",
    optimizer='adam',
    normalize=True,
    decay=0.1,
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
    return (np.asarray(lst) == lst[0]).all()


import tabulate

if __name__ == '__main__':
    df = pd.DataFrame(cfgs)
    print()

    res = []
    for j in range(df.shape[1]):
        if not is_all_same(df.iloc[:, j].tolist()): res.append(j)
    res = [df.columns[r] for r in res]
    df1 = df[res]
    df1.index = df1.logs_dir
    del df1['logs_dir']
    print(tabulate.tabulate(df1, headers="keys", tablefmt="pipe"))

# edict(
#     logs_dir='senet',
#     arch='resnet50', bottleneck='SEBottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.45, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='label',
#     dropout=0.25,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1,
#     pretrained=True, resume=None
# ),

# edict(
#     logs_dir='dop.tri.0.25',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=0,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=0.25,
#     pretrained=True, resume=None
# ),
# edict(
#     logs_dir='dop.tri.0.25.worker8',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=0.25,
#     pretrained=True, resume=None
# ),
# edict(
#     logs_dir='dop.tri.0.5',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=0,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=0.5,
#     pretrained=True, resume=None
# ),
# edict(
#     logs_dir='dop.tri.0.3',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=0,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=0.3,
#     pretrained=True, resume=None
# ),
# edict(
#     logs_dir='dop.tri.0.2',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=0,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=0.2,
#     pretrained=True, resume=None
# ),
# edict(
#     logs_dir='area.64',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.64, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1,
#     pretrained=True, resume=None
# ),
# edict(
#     logs_dir='area.96',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.96, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1,
#     pretrained=True, resume=None
# ),

#
# edict(
#     logs_dir='senet.cls.0.1',
#     arch='resnet50', bottleneck='SEBottleneck', dataset='cuhk03',
#     lr=1e-1, margin=0.45, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='label',
#     dropout=0.25,
#     cls_weight=1, tri_weight=0,
#     random_ratio=1,
#     pretrained=False, resume=None
# ),

# edict(
#     logs_dir='senet.cls.resume',
#     arch='resnet50', bottleneck='SEBottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.45, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='label',
#     dropout=0.25,
#     cls_weight=1, tri_weight=0,
#     random_ratio=1,
#     pretrained=False, resume='work/senet.cls/model_best.pth'
# ),

# edict(
#     logs_dir='senet.use_resnet',
#     arch='resnet50', bottleneck='SEBottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.45, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='label',
#     dropout=0.25,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1,
#     pretrained=True, resume=None
# ),

# edict(
#     logs_dir='dop.0.5.freeze',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.45, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=0,
#     dataset_mode='label',
#     dropout=0,
#     cls_weight=0.5,
#     tri_weight=0.5,
#     random_ratio=0.5,
# ),

# edict(
#     logs_dir='dop.0.5.worker.8',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.45, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='label',
#     dropout=0.25,
#     cls_weight=0.5,
#     tri_weight=0.5,
#     random_ratio=0.5,
# ),

# edict(
#     logs_dir='data.comb',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.45, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0.25,
#     cls_weight=0,
#     tri_weight=1,
#     random_ratio=1,
# ),
#
# edict(
#     logs_dir='loss.comb',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.45, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='label',
#     dropout=0.25,
#     cls_weight=0.5,
#     tri_weight=0.5,
#     random_ratio=1,
# ),
#
# edict(
#     logs_dir='loss.comb.0.1',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.45, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='label',
#     dropout=0.25,
#     cls_weight=0.1,
#     tri_weight=0.9,
#     random_ratio=1,
# ),
#
# edict(
#     logs_dir='no.dropout',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.45, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='label',
#     dropout=0,
#     cls_weight=0,
#     tri_weight=1,
#     random_ratio=1,
# ),
#
# edict(
#     logs_dir='dop.0.25',
#     arch='resnet50', bottleneck='Bottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.45, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=0,
#     dataset_mode='label',
#     dropout=0.25,
#     cls_weight=0.5,
#     tri_weight=0.5,
#     random_ratio=0.25,
# ),
#
