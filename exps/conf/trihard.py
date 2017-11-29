from lz import *
from easydict import EasyDict

cfgs = [
    EasyDict(dict(
        dataset='cuhk03',
        lr=1e-4,
        logs_dir='cuhk03',
        # log_at=np.concatenate([
        #     range(0, 100, 19),
        #     range(100, 150, 19),
        #     range(155, 165, 1)
        # ]),
        step=[100, 150, 160],
        epochs=165,
    )),
    # EasyDict(dict(
    #     dataset='market1501',
    #     lr=2e-3,
    #     logs_dir='market',
    #     # log_at=np.concatenate([
    #     #     range(0, 100, 19),
    #     #     range(100, 150, 19),
    #     #     range(155, 165, 1)
    #     # ]),
    #     step=[150, 180, 190],
    #     epochs=195,
    # ))

]

# for k, v in enumerate(cfgs):
#     if v.log_dir == '':
#         v.log_dir = '../work/'
#         for kk, vv in v.items():
#             # v.log_dir+= (str(vv)[:3] + '_' + str(hash(str(vv)))[:3])
#             # v.log_dir+='.'
#             v.log_dir += str(vv)
#         v.log_dir.rstrip('.')

base = EasyDict(
    dict(
        dbg=False,
        data_dir='/home/xinglu/.torch/data',
        restart=True,
        workers=8,
        split=0,
        height=256,
        width=128,
        combine_trainval=True,
        num_instances=4,
        # model
        evaluate=False,
        features=128,
        dropout=0,  # 0.5
        # loss
        margin=0.5,
        # optimizer
        lr=0.002,
        steps=[50, 100, 150],
        epochs=160,

        weight_decay=5e-4,
        # training configs
        resume='',
        start_save=0,
        seed=1,
        print_freq=5,
        # metric learning
        dist_metric='euclidean',

        branchs=8,
        branch_dim=64,
        use_global=False,

        loss='triplet',
        mode='hard',
        gpu=[0, ],
        pin_mem=True,
        log_start=False,
        log_middle=True,
        freeze='',
        # tuning
        dataset='cuhk03',
        batch_size=100,
        logs_dir='',
        arch='resnet50',
        embed="concat",
        optimizer='adam',
        normalize=True,
        num_classes=128,
        decay=0.1,
        config='',
        export_config=False,
        need_second=True,
        log_at=[100, 150],
    )
)

for k, v in enumerate(cfgs):
    v = dict_concat((base, v))
    cfgs[k] = EasyDict(v)
