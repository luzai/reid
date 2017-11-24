import argparse
import os.path as osp
from reid import models, datasets

def get_parser():

    parser = argparse.ArgumentParser(description="many kind loss classification")
    # data
    parser.add_argument('--restart', action='store_true', default=True)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation",
                        default=True)
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)  # 0.5
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.02,
                        help="learning rate of all parameters")
    parser.add_argument('--steps', type=list, default=[50, 100, 150])
    parser.add_argument('--epochs', type=int, default=160)

    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=5)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    home_dir = osp.expanduser('~') + '/.torch/'
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(home_dir, 'data'))

    parser.add_argument('--branchs', type=int, default=8)
    parser.add_argument('--branch_dim', type=int, default=64)
    parser.add_argument('--use_global', action='store_true', default=False)
    parser.add_argument('--loss', type=str, default='triplet',
                        choices=['triplet', 'tuple', 'softmax'])
    parser.add_argument('--mode', type=str, default='hard',
                        choices=['rand', 'hard', 'all', 'lift'])
    parser.add_argument('--gpu', type=list, default=[0, ])
    parser.add_argument('--pin_mem', action="store_true", default=True)
    parser.add_argument('--log_start',action='store_true')
    parser.add_argument('--log_middle',action='store_true')

    # tuning
    parser.add_argument('--freeze', action='store_true', default=False)
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())

    parser.add_argument('-b', '--batch-size', type=int, default=64)
    working_dir = osp.dirname(osp.abspath(__file__))

    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, ''))

    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--embed', type=str,default= "kron")
    parser.add_argument('--optimizer', type=str,default='sgd')
    parser.add_argument('--normalize', action='store_true',default=False)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--decay', type=float, default=0.5)
    parser.add_argument('--config', metavar='PATH', default='')
    parser.add_argument('--export-config', action='store_true', default=False)

    return parser