from lz import *
from easydict import EasyDict

cfgs = [
    EasyDict(dict(
        lr=3e-4,
        logs_dir='tribranch',
        arch='resnet50',
        dataset='cuhk03',
        dataset_val='cuhk03',
        batch_size=100, print_freq=1,
        num_instances=4,
        gpu=range(1),
        pin_mem=True,
        workers=8,
        branchs=8,
        branch_dim=64,
        global_dim=1024,
        num_classes=128,
        # resume='work.12.7/cuhk03/model_best.pth',
        evaluate=False,
        log_at=np.concatenate([
            range(10, 100, 49),
            range(100, 150, 19),
            range(155, 165, 1),
            # [0, 1, 2, 3, 4, 5, 6]
        ]),
        epochs=165,
    )),

]

base = EasyDict(
    dict(
        retrain=False,
        pretrained=True,
        hard_examples=True,
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

if __name__ == '__main__':
    # print(cfgs)
    for cfg in cfgs:
        print(cfg.logs_dir)
