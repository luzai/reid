import lz
from lz import *
from exps.tri_center import main as cmain
from exps.tri_xent import main as xmain

os.chdir(root_path + '/exps')

paths = glob.glob('work/*')
random.shuffle(paths)
for path in paths:
    try:
        # path = paths[0]
        if not osp.exists(path + '/conf.pkl'):
            continue
        args = pickle_load(path + '/conf.pkl')
        # if osp.exists(args.logs_dir + '/res.json'):
        #     continue
        # if args.dataset == 'market1501': continue
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
            res = cmain(args) # will release mem??
            # proc = mp.Process(target=cmain, args=(args,))
            # proc.start()
            # proc.join()
        elif 'xent' in args.loss:
            res = xmain(args)
            # proc = mp.Process(target=xmain, args=(args,))
            # proc.start()
            # proc.join()
        else:
            raise ValueError(args.loss)
        # print(res)
        # json_dump(res, args.logs_dir + '/res.json')
        # break
    except Exception as e:
        logging.error(e)
