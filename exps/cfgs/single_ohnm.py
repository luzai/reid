import sys

sys.path.insert(0, '/data1/xinglu/prj/open-reid')

from lz import *
from easydict import EasyDict as edict
import copy

no_proc = False
parallel = True
gpu_range = (0, 1, 2, 3)
cfgs = [
    # edict(
    #     logs_dir='11.xent.cu03lbl.bak',
    #     double=0, adv_inp=0, adv_fea=0, adv_inp_eps=0, adv_fea_eps=0,
    #     reg_mid_fea=[0., 0., 0., 0., 0.],  # x1, x2, x3, x4, x5
    #     reg_loss_wrt=[0, 0, 0, 0, 0, 0, ],  # input, x1, x2, x3,x4,x5
    #     # evaluate=True,
    #     # aux='l2_adv',
    #     dataset='cu03lbl',
    #     gpu=(1,), last_conv_stride=1, last_conv_dilation=1,
    #     gpu_fix=True,
    #     batch_size=64, num_instances=4, num_classes=128,
    #     dropout=0, loss='xent', tri_mode='adap',
    #     cls_weight=0, tri_weight=1, weight_dis_cent=0, weight_cent=0,
    #     random_ratio=1, lr_cent=0,
    #     gpu_range=gpu_range, lr_mult=1,
    #     push_scale=1., embed=None,
    #     margin='soft', margin2=0, margin3=0,
    #     # resume='/home/xinglu/work/reid/work.use/tri6.combine.2/model_best.pth',
    #     # resume='/home/xinglu/work/reid/work.use/tri6.combine.2/model_best.pth',
    #     # restart=True,
    #     epochs=35, steps=[15, 30], log_at=[2, 15, 30, 34, 35, 36], adv_eval=False,
    # ),

    # edict(
    #     logs_dir='10.mars.margin.-0.1',
    #     double=0, adv_inp=0, adv_fea=0, adv_inp_eps=0,
    #     reg_mid_fea=[0., 0., 0., 0., 0.],  # x1, x2, x3, x4, x5
    #     reg_loss_wrt=[0, 0, 0, 0, 0, 0, ],  # input, x1, x2, x3,x4,x5
    #     # evaluate=True,
    #     # aux='l2_adv',
    #     dataset='mars',
    #     gpu=(3,), last_conv_stride=2,
    #     # gpu_fix=True,
    #     batch_size=64, num_instances=4, num_classes=128, seq_len=15,
    #     dropout=0, loss='trivid', tri_mode='reg.a',
    #     cls_weight=0, tri_weight=1, weight_dis_cent=0, weight_cent=0,
    #     random_ratio=1, lr_cent=0,
    #     gpu_range=gpu_range, lr_mult=1,
    #     push_scale=1., embed=None,
    #     margin='soft', margin2=1, margin3=1.0, margin4=-.1,
    #     steps=[30, 50, ], epochs=55,
    #     # resume='/data1/xinglu/work/reid/work/10.mars.cont2/checkpoint.10.pth',
    #     # restart=True,
    #     workers=12, log_at=(0, 10, 20, 30, 40, 50, 59, 60, 61,),
    # ),

    edict(
        logs_dir='11.fx.margin.m2.1.1',
        double=0, adv_inp=0, adv_fea=0, adv_inp_eps=0,
        reg_mid_fea=[0., 0., 0., 0., 0.],  # x1, x2, x3, x4, x5
        reg_loss_wrt=[0, 0, 0, 0, 0, 0, ],  # input, x1, x2, x3,x4,x5
        # evaluate=True,
        # aux='l2_adv',
        dataset='mkt',
        gpu=(3,), last_conv_stride=2,
        # gpu_fix=True,
        batch_size=64, num_instances=4, num_classes=128,
        dropout=0, loss='tri', tri_mode='hard',
        cls_weight=0, tri_weight=1, weight_dis_cent=0, weight_cent=0,
        random_ratio=1, lr_cent=0,
        gpu_range=gpu_range, lr_mult=1,
        push_scale=1., embed=None,
        # margin='soft',
        margin=0.5,
        margin2=0, margin3=0, margin4=0,
        steps=[40, 60, ], epochs=65,
        log_at=[0, 30, 64, 65, 66],
        # resume='/data1/xinglu/work/reid/work/11.fx.margin.m2.1.1/checkpoint.64.pth',
        # evaluate=True,
        workers=12,
    ),

    # edict(
    #     logs_dir='11.fx.margin.m4.-0.1',
    #     double=0, adv_inp=0, adv_fea=0, adv_inp_eps=0,
    #     reg_mid_fea=[0., 0., 0., 0., 0.],  # x1, x2, x3, x4, x5
    #     reg_loss_wrt=[0, 0, 0, 0, 0, 0, ],  # input, x1, x2, x3,x4,x5
    #     # evaluate=True,
    #     # aux='l2_adv',
    #     dataset='cu03lbl',
    #     gpu=(3,), last_conv_stride=2,
    #     # gpu_fix=True,
    #     batch_size=128, num_instances=4, num_classes=128,
    #     dropout=0, loss='tri', tri_mode='reg.a',
    #     cls_weight=0, tri_weight=1, weight_dis_cent=0, weight_cent=0,
    #     random_ratio=1, lr_cent=0,
    #     gpu_range=gpu_range, lr_mult=1,
    #     push_scale=1., embed=None,
    #     margin='soft', margin2=1, margin3=1, margin4=-0.1,
    #     steps=[40, 60, ], epochs=65,
    #     log_at=[0, 30, 64, 65, 66],
    #     # resume='/data1/xinglu/work/reid/work/10.mars.cont2/checkpoint.10.pth',
    #     # restart=True,
    #     workers=12,
    # ),

]

# cfg = edict(
#     logs_dir='10.adv',
#     double=0, adv_inp=1, adv_fea=0, adv_inp_eps=0,
#     reg_mid_fea=[0., 0., 0., 0., 0.],  # x1, x2, x3, x4, x5
#     reg_loss_wrt=[0, 0, 0, 0, 0, 0, ],  # input, x1, x2, x3,x4,x5
#     # evaluate=True,
#     aux='l2_adv',  # l2_adv linf_adv nol_adv
#     dataset='cu03lbl',
#     gpu=(1,), last_conv_stride=2,
#     batch_size=64, num_instances=4, num_classes=128,
#     dropout=0, loss='tri', tri_mode='hard',
#     cls_weight=0, tri_weight=1, weight_dis_cent=0, weight_cent=0,
#     random_ratio=1, lr_cent=0,
#     gpu_range=gpu_range, lr_mult=1,
#     push_scale=1., embed=None,
#     margin='soft', margin2=1.0, margin3=1.0, margin4=0,
# )
#
# for m4 in [0]:
#     for ds in ['cu03lbl']:
#         for aux in ['l2_adv', 'linf_adv', 'nol_adv']:
#             for adv_inp_eps in [.3, 3e-2, 3e-3]:
#                 cfg_t = copy.deepcopy(cfg)
#                 cfg_t.margin4 = m4
#                 cfg_t.tri_mode = 'hard'
#                 cfg_t.dataset = ds
#                 cfg_t.aux = aux
#                 cfg_t.adv_inp_eps = adv_inp_eps
#                 cfg_t.logs_dir = f'{cfg.logs_dir}.{aux}.{adv_inp_eps}'
#                 cfgs.append(cfg_t)

# cfg = edict(
#     logs_dir='mkt.xent', double=1, adv_inp=1, adv_fea=0,
#     dataset='mkt', xent_smooth=True,
#     log_at=[0, 30, 64, 65, 66], lr_mult=10.,
#     # gpu_fix=True, gpu=(0, 1),
#     gpu=(0, 1, 2),
#     batch_size=128, num_instances=4, num_classes=128,
#     dropout=0, loss='xent', mode='',
#     cls_weight=0, tri_weight=1, weight_dis_cent=0, weight_cent=0,
#     random_ratio=1, lr_cent=0,
#     gpu_range=range(4),
#     push_scale=1., embed=None, margin=.5, margin2=1., margin3=1.,
# )
#
# for smooth, double, adv_inp in grid_iter(
#         [True, False],
#         [0, 1],
#         [0, 1],
# ):
#     cfg_t = copy.deepcopy(cfg)
#     cfg_t.double = double
#     cfg_t.adv_inp = adv_inp
#     cfg_t.xent_smooth = smooth
#     cfg_t.logs_dir = f'{cfg.logs_dir}.smth{smooth}.adv{adv_inp}.dbl.{double}'
#     cfgs.append(cfg_t)

base = edict(
    aux='',  # l2_adv linf_adv defaul: nol_adv; l1_grad default: l2_grad
    reg_mid_fea=[0., 0., 0., 0., 0.],
    amsgrad=False, freeze_bn=False,
    adv_eval=False, rerank=False,
    reg_loss_wrt=[0, 0, 0, 0, 0, 0],
    impr=0., cu03_classic=False,
    last_conv_stride=2,
    last_conv_dilation=1,
    double=0, adv_inp=0, adv_fea=0,
    adv_inp_eps=.3, adv_fea_eps=.3,
    optimizer_cent='adam', topk=5, test_best=True,
    push_scale=1., gpu_fix=False, test_batch_size=8,
    lr=3e-4, margin=0.5, area=(0.85, 1),
    margin2=0, margin3=0., margin4=0.,
    steps=[40, 60], epochs=65,
    arch='resnet50', block_name='Bottleneck', block_name2='Bottleneck', convop='nn.Conv2d',
    weight_dis_cent=0, vis=False,
    weight_cent=0, lr_cent=0.5, xent_smooth=False,
    adam_betas=(.9, .999), adam_eps=1e-8,
    lr_mult=1., fusion=None, eval_conf='market1501',
    cls_weight=0., random_ratio=1, tri_weight=1, num_deform=3, cls_pretrain=False,
    bs_steps=[], batch_size_l=[], num_instances_l=[],
    scale=(1,), translation=(0,), theta=(0,),
    hard_examples=False, has_npy=False, loss_div_weight=0,
    # double=0,  double conv is deprecated
    pretrained=True, dbg=False, data_dir='/home/xinglu/.torch/data',
    restart=True, workers=8, split=0, height=256, width=128,
    combine_trainval=True, num_instances=4,
    evaluate=False, dropout=0,
    seq_len=15, vid_pool='avg',
    log_at=None,
    weight_decay=5e-4, resume=None, start_save=0,
    seed=None, print_freq=3, dist_metric='euclidean',
    branchs=0, branch_dim=64, global_dim=1024, num_classes=128,
    loss='tri',
    tri_mode='hard', cent_mode='ccent.all.all',
    # mode='ccent.all.all', deprecated
    gpu=(0,), pin_mem=True, log_start=False, log_middle=True, gpu_range=range(4),
    # tuning
    dataset='mkt', dataset_val=None,
    batch_size=128, logs_dir='', embed=None,
    optimizer='adam', normalize=True, decay=0.1, run=0,
)

for ind, val in enumerate(cfgs):
    val = dict_update(base, val)
    cfgs[ind] = edict(val)

for ind, args in enumerate(cfgs):
    if args.evaluate is True \
            and args.resume is not None \
            and osp.exists(args.resume + '/conf.pkl'):
        resume = args.resume
        # logs_dir = args.logs_dir
        gpu_range = args.gpu_range
        gpu_fix = args.gpu_fix
        gpu = args.gpu
        bs = args.batch_size
        vis = args.vis
        args_ori = pickle_load(args.resume + '/conf.pkl')
        args = dict_update(args, args_ori)
        args = edict(args)
        args.resume = resume + '/model_best.pth'
        args.evaluate = True
        args.logs_dir = args.logs_dir.replace('work/', '') + '/eval'
        args.gpu_range = gpu_range
        args.gpu_fix = gpu_fix
        args.gpu = gpu
        args.batch_size = bs
        args.vis = vis
        cfgs[ind] = args

if __name__ == '__main__':
    import tabulate
    import pandas as pd


    def is_all_same(lst):
        lst = [lsti if not isinstance(lsti, np.ndarray) else lsti.tolist() for lsti in lst]
        res = [lsti == lst[0] for lsti in lst]
        try:
            return np.asarray(res).all()
        except Exception as e:
            print(e)


    df = pd.DataFrame(cfgs)
    if len(cfgs) == 1:
        print(df)
        exit(0)
    res = []
    for j in range(df.shape[1]):
        if not is_all_same(df.iloc[:, j].tolist()):
            res.append(j)
    res = [df.columns[r] for r in res]
    df1 = df[res]
    df1.index = df1.logs_dir
    # del df1['logs_dir']

    print(tabulate.tabulate(df1.sort_values(by='logs_dir'), headers="keys", tablefmt="pipe"))
