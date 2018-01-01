from lz import *
from easydict import EasyDict

cfgs = [
        EasyDict(dict(
            logs_dir='base',
            arch='resnet50',
            dataset='cuhk03',
            area=(0.85, 1),
            dataset_val='cuhk03',
            batch_size=128, print_freq=1, num_instances=4,
            gpu=range(2),
            double=512, loss_div_weight=0,
            translation=(0, 2 / 3),
            num_classes=128,
            log_at=np.concatenate([
                range(0, 100, 49),
                range(100, 150, 19),
                range(161, 165, 1),
                # range(165)
            ]),
            evaluate=False,
            epochs=165,
        )),
]

base = EasyDict(
    dict(
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
