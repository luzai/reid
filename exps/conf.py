from lz  import *
import  lz

working_dir =osp.dirname( osp.abspath(__file__) )
home_dir = ''
args_factory = {
    'gpus' : (0,1,2,3),
    'batch_size' : 160,

    'dataset': 'cuhk03' ,
    'workers': 32,
    'split':0 ,
    'height':256 ,
    'width':128,
    'combine_trainval':True ,
    'num_instances':4 ,
    'arch':'resnet50' ,
    'features':128,
    'dropout':0 ,
    'margin': 0.5,
    'lr':0.0002 ,
    'weight_decay':5e-4 ,
    'resume': '',
    'start_save':0 ,
    'seed':1 ,
    'print-freq':5 ,
    'dist_metric':'euclidean',
    'loss': 'triplet',
}
args = args_factory

def dict2obj():
    pass

for k,v in args_factory.iteritems():
    pass



args.logs_dir += ('.' + args.loss)
dbg = False
if dbg:
    lz.init_dev((3,))
    args.epochs = 2
    args.batch_size = 8
    args.logs_dir = args.logs_dir + '.dbg'
lz.mkdir_p(args.logs_dir, delete=True)
lz.write_json(vars(args), args.logs_dir + '/conf.json')
for k, v in vars(args).iteritems():
    print('{}: {}'.format(k, v))
if args.loss == 'softmax':
    args.num_instances = None
main(args)