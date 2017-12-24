from lz import *
from easydict import EasyDict

cfgs = [
    EasyDict(dict(
        lr=3e-4,
        logs_dir='double.dbg',
        arch='resnet50',
        dataset='cuhk03',
        area=(0.85, 1),
        dataset_val='cuhk03',
        batch_size=100, print_freq=1, num_instances=4,
        gpu=range(1),
        has_npy=False, double=True, loss_div_weight=1e-2,
        global_dim=1024,
        num_classes=128,
        log_at=np.concatenate([
            range(0, 100, 49),
            range(100, 150, 19),
            range(161, 165, 1),
        ]),
        # evaluate=True,
        # resume='work/res50.doubly/model_best.pth',
        epochs=165,
    )),
    # EasyDict(dict(
    #         lr=3e-4,
    #         logs_dir='fuck.all.data.long.cont',
    #         arch='resnet50',
    #         dataset=['cuhk03', 'dukemtmc', 'market1501'],
    #         dataset_val='cuhk03',
    #         batch_size=128, print_freq=200,
    #         num_instances=4,
    #         gpu=range(1),
    #         has_npy=False, double=False,
    #         branchs=0,
    #         branch_dim=64,
    #         global_dim=1024,
    #         num_classes=128,
    #         resume='work/fuck.all.data.long/model_best.pth',
    #         restart=False,
    #         log_at=np.concatenate([
    #             range(0, 185, 1),
    #         ]),
    #         epochs=185,
    #     )),
    # EasyDict(dict(
    #     lr=3e-4,
    #     logs_dir='resolution.224',
    #     arch='resnet50',
    #     height=224,
    #     width=224,
    #     dataset='cuhk03',
    #     dataset_val='cuhk03',
    #     batch_size=128, print_freq=1,
    #     num_instances=4,
    #     gpu=range(2),
    #     has_npy=False,
    #     branchs=0,
    #     branch_dim=64,
    #     global_dim=1024,
    #     num_classes=128,
    #     log_at=np.concatenate([
    #         range(0, 90, 49),
    #         range(100, 135, 19),
    #         range(155, 165, 1),
    #     ]),
    #     epochs=165,
    # )),
    # EasyDict(dict(
    #     lr=3e-4,
    #     logs_dir='doubly.3.combine',
    #     arch='resnet50',
    #     dataset='cuhk03',
    #     dataset_mode='combine',
    #     area=(0.85, 1),
    #     dataset_val='cuhk03',
    #     batch_size=128, print_freq=1,
    #     num_instances=4,
    #     gpu=range(2),
    #     has_npy=False,double=True,
    #     branchs=0,
    #     branch_dim=64,
    #     global_dim=1024,
    #     num_classes=128,
    #     log_at=np.concatenate([
    #         range(0, 100, 49),
    #         range(100, 150, 19),
    #         range(155, 165, 1),
    #     ]),
    #     # evaluate=True,
    #     # resume='work/res50.doubly/model_best.pth',
    #     epochs=165,
    # )),
    # EasyDict(dict(
    #     lr=3e-4,
    #     logs_dir='res50.256.detect.0.64.2.3',
    #     arch='resnet50',
    #     dataset='cuhk03',
    #     dataset_mode='detect',
    #     area=(0.64, 1),
    #     dataset_val='cuhk03',
    #     batch_size=256, print_freq=1,
    #     num_instances=4,
    #     gpu=range(2),
    #     has_npy=False,
    #     branchs=0,
    #     branch_dim=64,
    #     global_dim=1024,
    #     num_classes=128,
    #     log_at=np.concatenate([
    #         range(0, 100, 49),
    #         range(100, 150, 19),
    #         range(155, 165, 1),
    #     ]),
    #     epochs=165,
    # )),
    # EasyDict(dict(
    #     lr=3e-4,
    #     logs_dir='res50.256.label.0.64',
    #     arch='resnet50',
    #     dataset='cuhk03',
    #     dataset_mode='label',
    #     area=(0.64, 1),
    #     dataset_val='cuhk03',
    #     batch_size=256, print_freq=1,
    #     num_instances=4,
    #     gpu=range(2),
    #     has_npy=False,
    #     branchs=0,
    #     branch_dim=64,
    #     global_dim=1024,
    #     num_classes=128,
    #     log_at=np.concatenate([
    #         range(0, 100, 49),
    #         range(100, 150, 19),
    #         range(155, 165, 1),
    #     ]),
    #     epochs=165,
    # )),
    # EasyDict(dict(
    #     lr=3e-4,
    #     logs_dir='res50.256.label.0.85',
    #     arch='resnet50',
    #     dataset='cuhk03',
    #     dataset_mode='label',
    #     area=(0.85, 1),
    #     dataset_val='cuhk03',
    #     batch_size=256, print_freq=1,
    #     num_instances=4,
    #     gpu=range(2),
    #     has_npy=False,
    #     branchs=0,
    #     branch_dim=64,
    #     global_dim=1024,
    #     num_classes=128,
    #     log_at=np.concatenate([
    #         range(0, 100, 49),
    #         range(100, 150, 19),
    #         range(155, 165, 1),
    #     ]),
    #     epochs=165,
    # )),
    # EasyDict(dict(
    #     lr=3e-4,
    #     logs_dir='res50.bs.256',
    #     arch='resnet50',
    #     dataset='cuhk03',
    #     dataset_val='cuhk03',
    #     batch_size=256, print_freq=1,
    #     num_instances=4,
    #     gpu=range(2),
    #     has_npy=False,
    #     branchs=0,
    #     branch_dim=64,
    #     global_dim=1024,
    #     num_classes=128,
    #     # evaluate=True,
    #     # resume='work/res50.base/model_best.pth',
    #     log_at=np.concatenate([
    #         range(0, 100, 49),
    #         range(100, 150, 19),
    #         range(155, 165, 1),
    #     ]),
    #     epochs=165,
    # )),
    # EasyDict(dict(
    #     lr=3e-4,
    #     logs_dir='res50.bs.128.lc',
    #     arch='resnet50',
    #     dataset='cuhk03',
    #     dataset_val='cuhk03',
    #     batch_size=128, print_freq=1,
    #     num_instances=4,
    #     gpu=range(1),
    #     has_npy=False,
    #     branchs=8,
    #     branch_dim=64,
    #     global_dim=1024,
    #     num_classes=128,
    #     # evaluate=True,
    #     # resume='work/res50.base/model_best.pth',
    #     log_at=np.concatenate([
    #         range(0, 100, 49),
    #         range(100, 150, 19),
    #         range(155, 165, 1),
    #     ]),
    #     epochs=165,
    # )),
]

base = EasyDict(
    dict(
        has_npy=False, double=False, loss_div_weight=0,
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
        # tuning
        dataset='market1501',
        dataset_mode='combine',
        area=(0.85, 1),
        dataset_val='market1501',
        batch_size=100,
        logs_dir='',
        arch='resnet50',
        embed="concat",
        optimizer='adam',
        normalize=True,
        decay=0.1,
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
