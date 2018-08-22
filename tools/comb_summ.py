import lz
from lz import *

fpath = work_path + 'reid/work.8.1'
fpath = osp.abspath(fpath)


def clean_empty():
    paths = glob.glob(fpath)
    for path in paths:
        size = os.stat(path).st_size
        # print(size)
        if size < 67:
            print(path, size)
            # shell(f'trash {path}')


clean_empty()
os.chdir((fpath))
paths = glob.glob('*')
assert len(paths) > 0, paths
# random.shuffle(paths)
res_dict = {}
cfg_dict = {}
cfgs = []
for path in paths:
    # name = args.logs_dir.split('/')[-1]
    name = path.split('/')[-1]
    print(name)
    # if 'only' in name: continue
    # if 'duke' not in name: continue
    # if 'cfisher' not in name or 'cu'  in name: continue
    # if 'tuning' not in name: continue
    # if not 'cu01' in name and not 'cuhk01' in name: continue

    assert osp.exists(path + '/conf.pkl'), path
    # if not osp.exists(path + '/conf.pkl'):
    #     continue
    args = pickle_load(path + '/conf.pkl')
    # if osp.exists(args.logs_dir + '/eval/res.json'):
    # res_path = args.logs_dir + '/eval/res.json'
    # else:
    args.logs_dir = fpath + '/' + osp.basename(osp.abspath(args.logs_dir))
    res_path = args.logs_dir + '/res.json'

    if not osp.exists(res_path):
        res, _ = shell(f'tail {osp.dirname(res_path)+"/log.txt"} -n 1')
        try:
            import json

            json.loads(res)
        except:
            continue
    else:
        # print(args.logs_dir)
        res = json_load(res_path)
    # print(args, res)
    res_dict[name] = res
    cfg_dict[name] = args
    cfgs.append(args)

df_acc = pd.DataFrame(res_dict).T
df_acc *= 100


def is_all_same(lst):
    lst = [lsti if not isinstance(lsti, np.ndarray) else lsti.tolist() for lsti in lst]
    res = [lsti == lst[0] for lsti in lst]
    # try:
    return np.asarray(res).all()
    # except Exception as e:
    #     print(e)


df = pd.DataFrame(cfgs)
if len(cfgs) == 1:
    raise ValueError
res = []
for j in range(df.shape[1]):
    if not is_all_same(df.iloc[:, j].tolist()):
        res.append(j)
res = [df.columns[r] for r in res]
df_cfg = df[res]

print(df_cfg.head())
df_cfg.index = df_cfg.logs_dir.str.replace(fpath, '').str.replace(osp.abspath(work_path + 'reid/work'), '').str.strip(
    '/')
print(df_cfg.head())
print(df_acc.head())
df_final = pd.concat((  df_cfg, df_acc), axis=1)
df_final = df_final[['top-1', 'top-5', 'top-10', 'mAP'] + list(df_final.keys())]
df_final.to_excel(work_path + 't.xlsx')
exit(0)


def f1(x):
    return r'%.2f' % x


t = df_acc[['top-1',
            'mAP',
            # 'top-5', 'top-10',
            ]]
print(t.to_latex(formatters=[f1, ] * 3))
# print(df[['top-1.rk',
#           # 'mAP.rk',
#           'top-5.rk','top-10.rk',
#           ]].to_latex(formatters=[f1, ] * 3))
# print(t)
# print(df[['top-1.rk',
#           # 'mAP.rk',
#           'top-5.rk','top-10.rk',
#           ]])
