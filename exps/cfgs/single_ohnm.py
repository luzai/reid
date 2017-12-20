from lz import *
from easydict import EasyDict

cfgs = [
    # EasyDict(dict(
    #     lr=3e-4,
    #     # logs_dir='res50.bs.512.lc.64.8.inst.8',
    #     arch='resnet50',
    #     dataset='cuhk03',
    #     dataset_val='cuhk03',
    #     batch_size=128, print_freq=1,
    #     num_instances=8,
    #     gpu=range(1),
    #     has_npy=False,
    #     branchs=8,
    #     branch_dim=64,
    #     global_dim=1024,
    #     num_classes=128,
    #     evaluate=True,
    #     resume='work/res50.bs.512.lc.64.8.inst.8/model_best.pth',
    #     log_at=np.concatenate([
    #         range(0, 100, 49),
    #         range(100, 150, 19),
    #         range(155, 165, 1),
    #     ]),
    #     epochs=165,
    # )),
    EasyDict(dict(
        lr=3e-4,
        # logs_dir='res34.bs.1024.lc.64.8.int16',
        arch='resnet34',
        dataset='cuhk03',
        dataset_val='cuhk03',
        batch_size=128, print_freq=1,
        num_instances=16,
        gpu=range(1),
        has_npy=False,
        branchs=8,
        branch_dim=64,
        global_dim=1024,
        num_classes=128,
        evaluate=True,
        resume='work/res34.bs.1024.lc.64.8.int16/model_best.pth',
        log_at=np.concatenate([
            range(0, 100, 49),
            range(100, 150, 19),
            range(155, 165, 1),
        ]),
        epochs=165,
    )),
]

base = EasyDict(
    dict(
        has_npy=False,
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

        lr=3e-4,

        # steps=[150, 200, 210],
        # epochs=220,
        # log_at=np.concatenate([
        #     range(0, 150, 9),
        #     range(150, 200, 5),
        #     range(200, 220, 1)
        # ]),
        steps=[100, 150, 160],
        epochs=165,
        log_at=np.concatenate([
            range(0, 100, 49),
            range(100, 150, 19),
            range(155, 165, 1),
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
        dataset_val='market1501',
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


def format_cfg(cfg):
    if cfg.gpu is not None:
        cfg.pin_mem = True
        cfg.workers = len(cfg.gpu) * 8
    else:
        cfg.pin_mem = False
        cfg.workers = 4


if __name__ == '__main__':
    # print(cfgs)
    for cfg in cfgs:
        print(cfg.logs_dir)
