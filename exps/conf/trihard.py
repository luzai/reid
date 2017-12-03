from lz import *
from easydict import EasyDict

cfgs = [
EasyDict(dict(
        lr=2e-4,
        logs_dir='global',

        batch_size=100,
        gpu=range(1),
        branchs=0,
        branch_dim=128,
        global_dim=1024,
        num_classes=128,
    )),
    EasyDict(dict(
        lr=2e-4,
        logs_dir='gllo.8.128',

        batch_size=100,
        gpu=range(1),
        branchs=8,
        branch_dim=128,
        global_dim=1024,
        num_classes=128,
    )),
    EasyDict(dict(
        lr=2e-4,
        logs_dir='gllo.4.256',

        batch_size=100,
        gpu=range(1),
        branchs=4,
        branch_dim=256,
        global_dim=1024,
        num_classes=128,
    )),
    EasyDict(dict(
        lr=2e-4,
        logs_dir='local.8.128',

        batch_size=100,
        gpu=range(1),
        branchs=8,
        branch_dim=128,
        global_dim=0,
        num_classes=128,
    )),
    EasyDict(dict(
        lr=2e-4,
        logs_dir='local.8.256',

        batch_size=100,
        gpu=range(1),
        branchs=8,
        branch_dim=256,
        global_dim=0,
        num_classes=128,
    )),
    EasyDict(dict(
        lr=2e-4,
        logs_dir='local.4.256',

        batch_size=100,
        gpu=range(1),
        branchs=4,
        branch_dim=256,
        global_dim=0,
        num_classes=128,
    )),
    EasyDict(dict(
        lr=2e-4,
        logs_dir='local.4.512',

        batch_size=100,
        gpu=range(1),
        branchs=4,
        branch_dim=512,
        global_dim=0,
        num_classes=128,
    )),

]

cfgs_done = [
    EasyDict(dict(
        dataset='market1501',
        arch='inception_v3',
        optimizer='adam',
        pretrained=False,
        lr=1.5e-4,
        logs_dir='inceptionv3.pretrainfalse',
        steps=[150, 200, 210],
        epochs=215,
        log_at=np.concatenate([
            range(0, 100, 49),
            range(100, 210, 100),
            range(212, 215, 1)
        ]),
        batch_size=100,
        gpu=range(8),
        height=256, width=128,
    )),
    EasyDict(dict(
        dataset='market1501',
        arch='inception',
        optimizer='adam',
        lr=1.5e-4,
        logs_dir='inception',
        steps=[150, 200, 210],
        epochs=215,
        log_at=np.concatenate([
            range(0, 100, 49),
            range(100, 210, 100),
            range(212, 215, 1)
        ]),
        batch_size=100,
        gpu=[0, 1, 2, 3],
        height=256, width=128,
    )),

    EasyDict(dict(
        dataset=['market1501', 'cuhk03'],
        dataset_val='cuhk03',
        optimizer='adam',
        lr=2.1e-4,
        logs_dir='cuhk03.external',
        steps=[150, 200, 210],
        epochs=215,
        log_at=np.concatenate([
            range(0, 100, 49),
            range(100, 210, 100),
            range(212, 215, 1)
        ]),
    )),
    EasyDict(dict(
        dataset=['market1501', 'cuhk03'],
        dataset_val='market1501',
        optimizer='adam',
        lr=2.1e-4,
        logs_dir='mk.train.dbg',
        steps=[150, 200, 210],
        epochs=215,
        log_at=np.concatenate([
            range(0, 100, 49),
            range(100, 210, 100),
            range(212, 215, 1)
        ]),
    )),

]

# meta = EasyDict(dict(
#     dataset='cuhk03',
#     optimizer='adam',
#     lr=0.01,
#     logs_dir='',
#     steps=[100, 150, 160],
#     epochs=5,
# ))
#
# for lr in np.logspace(-4, -3, 4):
#     meta.lr = lr
#     meta.logs_dir = 'cuhk03.adam.' + str(float(lr))
#     cfgs.append(meta.copy())
#
# meta = EasyDict(dict(
#     dataset='market1501',
#     optimizer='adam',
#     steps=[100, 150, 160],
#     epochs=5,
# ))
#
# for lr in np.logspace(-4, -3, 4):
#     meta.lr = lr
#     meta.logs_dir = 'mk.adam.' + str(float(lr))
#     cfgs.append(meta.copy())

base = EasyDict(
    dict(
        pretrained=True,
        dbg=False,
        data_dir='/home/xinglu/.torch/data',
        restart=True,
        workers=8, split=0, height=256, width=128, combine_trainval=True,
        num_instances=4,
        # model
        evaluate=False,
        dropout=0,  # 0.5
        # loss
        margin=0.5,
        # optimizer

        lr=2e-4,
        steps=[150, 200, 210],
        epochs=220,
        log_at=np.concatenate([
            range(0, 150, 9),
            range(150, 200, 5),
            range(200, 220, 1)
        ]),

        weight_decay=5e-4,
        resume='',
        start_save=0,
        seed=1, print_freq=5, dist_metric='euclidean',

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
        freeze='',
        # tuning
        dataset='market1501',
        dataset_val='',
        batch_size=100,
        logs_dir='',
        arch='resnet50',
        embed="concat",
        optimizer='adam',
        normalize=True,
        decay=0.1,
        export_config=False,
        need_second=True,
    )
)

for k, v in enumerate(cfgs):
    v = dict_update(base, v)
    cfgs[k] = EasyDict(v)

if __name__ == '__main__':
    # print(cfgs)
    for cfg in cfgs:
        print(cfg.logs_dir)
