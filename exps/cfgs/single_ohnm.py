import sys

sys.path.insert(0, '/data1/xinglu/prj/luzai-tool')
sys.path.insert(0, '/data1/xinglu/prj/open-reid')

from lz import *

cfgs = [
    # edict(
    #     logs_dir='msmt17.res.long',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='msmt17', dataset_val='msmt17', eval_conf='market1501',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=(0,), num_classes=128,
    #     steps=[80, 120], epochs=125,
    #     workers=8,
    #     dropout=0, loss='triplet',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, fusion=None,
    #     log_at=[124, 125],
    #     evaluate=True,
    #     resume='/data1/xinglu/prj/open-reid/exps/work/msmt17.res.2/model_best.pth'
    # ),

    # edict(
    #     logs_dir='cuhk03label.res.center',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     # dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     # dataset='msmt17', dataset_val='msmt17', eval_conf='market1501',
    #     dataset='market1501', dataset_val='market1501', eval_conf='market1501',
    #     # dataset='mars', dataset_val='mars', eval_conf='market1501',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='label',
    #     dropout=0, loss='center',
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
    #     workers=8,
    #     dataset_mode='label',
    #     dropout=0, loss='quin',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, fusion=None,
    # ),
    #
    # edict(
    #     logs_dir='cuhk03label.res.quin.dop',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='label',
    #     dropout=0, loss='quin',
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=0.5, fusion=None,
    # ),

    # edict(
    #     logs_dir='cuhk03label.se',
    #     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='label',
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, fusion=None,
    #     evaluate=True, resume='work.3.8/cuhk03label.se/model_best.pth'
    # ),

    # edict(
    #     logs_dir='cuhk03detect.se.concat',
    #     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='detect',
    #     dropout=0,
    #     cls_weight=0.5, tri_weight=1,
    #     random_ratio=1, fusion='concat',
    #     evaluate=True,
    #     resume='/data1/xinglu/prj/open-reid/exps/work.3.8/cuhk03detect.se.concat/model_best.pth'
    # ),

    # edict(
    #     logs_dir='cuhk03label.se.sum.dop',
    #     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='label',
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=0.5, fusion='sum',
    #     # evaluate=True,
    #     # resume='work/cuhk03label.se.concat.dop/model_best.pth',
    # ),

    # edict(
    #     logs_dir='cuhk03detect.res',
    #     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='detect',
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, fusion=None,
    #     resume = '/data1/xinglu/prj/open-reid/exps/work.3.8/cuhk03detect.res/model_best.pth', evaluate =True
    # ),

    # edict(
    #     logs_dir='cuhk03detect.se',
    #     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='detect',
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, fusion=None,
    #     evaluate=True,
    #     resume='/data1/xinglu/prj/open-reid/exps/work.3.8/cuhk03detect.se/model_best.pth',
    # ),

    # edict(
    #     logs_dir='cuhk03detect.se.sum',
    #     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='detect',
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, fusion='sum',
    # ),
    #
    # edict(
    #     logs_dir='cuhk03detect.se.sum.dop',
    #     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
    #     dataset='cuhk03', dataset_val='cuhk03', eval_conf='cuhk03',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dataset_mode='detect',
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=0.5, fusion='sum',
    # ),

    edict(
        logs_dir='market1501.res.center',
        arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
        dataset='market1501', dataset_val='market1501', eval_conf='market1501',
        lr=3e-4, margin=0.5, area=(0.85, 1),
        batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
        steps=[40, 60], epochs=65,
        workers=8, dropout=0,
        cls_weight=0, tri_weight=1,
        loss='center',
        random_ratio=1, fusion=None,
        # evaluate=True, resume='/data1/xinglu/prj/open-reid/exps/work.3.8/market1501.res/model_best.pth'
    ),

    # edict(
    #     logs_dir='market1501.se',
    #     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
    #     dataset='market1501', dataset_val='market1501', eval_conf='market1501',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, fusion=None,
    # ),
    # edict(
    #     logs_dir='market1501.se.sum',
    #     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
    #     dataset='market1501', dataset_val='market1501', eval_conf='market1501',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=1, fusion='sum',
    # ),
    # edict(
    #     logs_dir='market1501.se.sum.dop',
    #     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
    #     dataset='market1501', dataset_val='market1501', eval_conf='market1501',
    #     lr=3e-4, margin=0.5, area=(0.85, 1),
    #     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
    #     steps=[40, 60], epochs=65,
    #     workers=8,
    #     dropout=0,
    #     cls_weight=0, tri_weight=1,
    #     random_ratio=0.5, fusion='sum',
    # ),

]

# cfgs = [cfgs[-1]]

base = edict(
    weight_cent = 5e-4,  lr_cent = 0.5,
    lr_mult=0.1, fusion=None, eval_conf='cuhk03',
    cls_weight=0., random_ratio=1, tri_weight=1, num_deform=3, cls_pretrain=False,
    bs_steps=[], batch_size_l=[], num_instances_l=[],
    block_name='Bottleneck', block_name2='Bottleneck', convop='nn.Conv2d',
    scale=(1,), translation=(0,), theta=(0,),
    hard_examples=False, has_npy=False, double=0, loss_div_weight=0,
    pretrained=True, dbg=False, data_dir='/home/xinglu/.torch/data',
    restart=True, workers=8, split=0, height=256, width=128,
    combine_trainval=True, num_instances=4,
    # model
    evaluate=False, dropout=0, margin=0.45,
    # optimizer
    lr=3e-4, steps=[100, 150, 160], epochs=165,
    log_at=np.concatenate([
        range(0, 640, 21),
    ]),
    weight_decay=5e-4, resume=None, start_save=0,
    seed=1, print_freq=3, dist_metric='euclidean',
    branchs=0, branch_dim=64, global_dim=1024, num_classes=128,
    loss='triplet', mode='hard',
    gpu=[0, ], pin_mem=True, log_start=False, log_middle=True,
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
#     logs_dir='se.unet.deform',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEUnetDeformBottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, num_deform=1,
# ),

# edict(
#     logs_dir='sedeform.1.lr_mult.0',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEDeformBottleneck',
#     dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, num_deform=1, lr_mult=0,
# ),
# edict(
#     logs_dir='sedeform.1',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEDeformBottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, num_deform=1,
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
#     random_ratio=1,
#     # cls_pretrain=True,
# ),
#
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
#
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
#     logs_dir='cuhk01.res',
#     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
#     dataset='cuhk01', dataset_val='cuhk01',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, fusion=None,
# ),
# edict(
#     logs_dir='cuhk01.se.concat',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
#     dataset='cuhk01', dataset_val='cuhk01',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0.35,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, fusion='concat',
# ),
# edict(
#     logs_dir='cuhk01.res.dop',
#     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
#     dataset='cuhk01', dataset_val='cuhk01',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=0.5, fusion=None,
# ),

# edict(
#     logs_dir='dukemtmc.res',
#     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
#     dataset='dukemtmc', dataset_val='dukemtmc',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, fusion=None,
# ),
# edict(
#     logs_dir='dukemtmc.se.concat',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
#     dataset='dukemtmc', dataset_val='dukemtmc',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0.35,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, fusion='concat',
# ),
# edict(
#     logs_dir='dukemtmc.res.dop',
#     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
#     dataset='dukemtmc', dataset_val='dukemtmc',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=0.5, fusion=None,
# ),

# edict(
#     logs_dir='viper.res',
#     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
#     dataset='viper', dataset_val='viper',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, fusion=None,
# ),
# edict(
#     logs_dir='viper.se.concat',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
#     dataset='viper', dataset_val='viper',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0.35,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, fusion='concat',
# ),
# edict(
#     logs_dir='viper.res.dop',
#     arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck',
#     dataset='viper', dataset_val='viper',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=0.5, fusion=None,
# ),

# edict(
#     logs_dir='senet.sum.new_avg',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
#     dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, fusion='sum',
# ),
# edict(
#     logs_dir='senet.concat.drop.0.35',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
#     dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0.35,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, fusion='concat',
# ),
# edict(
#     logs_dir='senet',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
#     dataset='cuhk03',
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
#     logs_dir='senet.init_senet.dop.0.5.cdm',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
#     dataset='cdm',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=0,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=0.5,
# ),
# edict(
#     logs_dir='senet.init_senet.dop.0.5.cdm.worker8.longer.2',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
#     dataset='cdm',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[80, 120], epochs=125,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=0.5,
# ),

# edict(
#     logs_dir='senet.init_senet.dop.0.5.cdm.worker8',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
#     dataset='cdm',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=0.5,
# ),

# edict(
#     logs_dir='senet.init_senet.cdm.worker8',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck',
#     dataset='cdm',
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
#     logs_dir='unet.deform',
#     arch='resnet50', block_name='SEBottleneck', block_name2='UnetDeformBottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=1, num_deform=1,
# ),

# edict(
#     logs_dir='senet.init_senet.dop.rand.0.25.worker8',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=0.25,
# ),

# edict(
#     logs_dir='senet.init_senet.dop.worker8',
#     arch='resnet50', block_name='SEBottleneck', block_name2='SEBottleneck', dataset='cuhk03',
#     lr=3e-4, margin=0.5, area=(0.85, 1),
#     dataset_val='cuhk03', batch_size=128, num_instances=4, gpu=range(1), num_classes=128,
#     steps=[40, 60], epochs=65,
#     workers=8,
#     dataset_mode='combine',
#     dropout=0,
#     cls_weight=0, tri_weight=1,
#     random_ratio=0.25,
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
