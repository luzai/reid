import lz
from lz import *
from exps.tri_center import main as cmain
from exps.tri_xent import main as xmain

os.chdir(root_path + '/exps')

paths = glob.glob('work/*')
random.shuffle(paths)
for path in paths:
    print('now path',path )
    try:
        # path = paths[0]
        if not osp.exists(path + '/conf.pkl'):
            continue
        args = pickle_load(path + '/conf.pkl')
        if args.logs_dir != path:
            print('not same', args.logs_dir)
        args.logs_dir = path + '/eval'
        args.gpu_range = range(4)
        # if osp.exists(args.logs_dir + '/res.json'):     continue
        # if args.dataset == 'market1501': continue
        # if args.logs_dir != 'work/tri_cu01_search.3e-04.0.5.1e-08.32': continue
        # if 'multis' not in path: continue
        if not 'cu01' in path : continue

        if args.dataset == 'cu03det' or 'det' in path:
            args.dataset = 'cuhk03'
            args.dataset_val = 'cuhk03'
            args.dataset_mode = 'detect'
            args.eval_conf = 'cuhk03'
        elif args.dataset == 'cu03lbl' or 'lbl' in path:
            args.dataset = 'cuhk03'
            args.dataset_val = 'cuhk03'
            args.dataset_mode = 'label'
            args.eval_conf = 'cuhk03'
        elif args.dataset == 'mkt' or args.dataset == 'market' or args.dataset == 'market1501':
            args.dataset = 'market1501'
            args.dataset_val = 'market1501'
            args.eval_conf = 'market1501'
        elif args.dataset == 'msmt':
            args.dataset = 'msmt17'
            args.dataset_val = 'market1501'
            args.eval_conf = 'market1501'
        elif args.dataset == 'cdm':
            args.dataset = 'cdm'
            args.dataset_val = 'market1501'
            args.eval_conf = 'market1501'
        elif args.dataset == 'viper':
            args.dataset = 'viper'
            args.dataset_val = 'viper'
            args.eval_conf = 'market1501'
        elif 'hard' in path:
            args.dataset = 'cuhk01'
            args.dataset_val = 'cuhk01'
            args.eval_conf = 'cuhk03'
            args.dataset_mode = 'hard'
        elif  'easy' in path:
            args.dataset = 'cuhk01'
            args.dataset_val = 'cuhk01'
            args.eval_conf = 'cuhk03'
            # args.eval_conf = 'market1501'
            args.dataset_mode = 'easy'
        elif args.dataset == 'dukemtmc':
            args.dataset = 'dukemtmc'
            args.dataset_val = 'dukemtmc'
            args.eval_conf = 'market1501'
        elif args.dataset == 'cuhk01':
            args.dataset = 'cuhk01'
            args.dataset_val = 'cuhk01'
            args.eval_conf = 'cuhk03'
            args.dataset_mode = 'hard'
        else:
            # raise ValueError(f'dataset ... {args.dataset}')
            logging.error(f'pls see {path}')

        args.gpu = lz.get_dev(n=len(args.gpu),
                              ok=range(4),
                              mem=[0.12, 0.05], sleep=32.3)

        args.evaluate = True
        args.resume = path + '/model_best.pth'
        if not osp.exists(args.resume):
            logging.info(args.logs_dir, 'shoule delete')
            # rm(args.logs_dir)
            continue
        if 'cent' in args.loss:
            # res = cmain(args)  # will release mem??
            proc = mp.Process(target=cmain, args=(args,))
            proc.start()
            proc.join()
        elif 'xent' in args.loss:
            # res = xmain(args)
            proc = mp.Process(target=xmain, args=(args,))
            proc.start()
            proc.join()
        else:
            raise ValueError(args.loss)
        # print(res)
        # json_dump(res, args.logs_dir + '/res.json')
        # break
    except Exception as e:
        logging.error(e)
        logging.error(f'exception path {path}')
