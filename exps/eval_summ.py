import lz
from lz import *
from exps.tri_center_xent import main

os.chdir(root_path + '/exps')

paths = glob.glob('work/*')
random.shuffle(paths)
for path in paths:
    print('now path', path)
    try:
        # path = paths[0]
        if not osp.exists(path + '/conf.pkl'):
            continue
        args = pickle_load(path + '/conf.pkl')
        if args.logs_dir != path:
            print('not same', args.logs_dir)
        args.logs_dir = path + '/eval'
        args.gpu_range = range(4)
        if osp.exists(args.logs_dir + '/res.json'): continue
        if 'mkt' not in  args.logs_dir: continue
        args.dataset_val = args.dataset
        args.eval_conf = 'market1501'

        args.gpu = lz.get_dev(n=len(args.gpu),
                              ok=range(4),
                              mem=[0.12, 0.05], sleep=32.3)

        args.evaluate = True
        args.resume = path + '/model_best.pth'
        if not osp.exists(args.resume):
            logging.info(args.logs_dir, 'shoule delete')
            # rm(args.logs_dir)
            continue

        # res = cmain(args)  # will not release mem
        proc = mp.Process(target=main, args=(args,))
        proc.start()
        proc.join()

        # print(res)
        # json_dump(res, args.logs_dir + '/res.json')
        # break
    except Exception as e:
        logging.error(e)
        logging.error(f'exception path {path}')
