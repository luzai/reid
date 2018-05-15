import lz
from lz import *
from exps.tri_center import main as cmain
from exps.tri_xent import main as xmain


def clean_empty():
    paths = glob.glob('work/*/*.amax')
    for path in paths:
        size = os.stat(path).st_size
        # print(size)
        if size < 67:
            print(path, size)
            # shell(f'trash {path}')


os.chdir(root_path + '/exps')
paths = glob.glob('work/*')
# random.shuffle(paths)
res_dict = {}
cfg_dict = {}
for path in paths:
    # name = args.logs_dir.split('/')[-1]
    name = path.split('/')[-1]
    if 'only' in name: continue
    # if not 'multis.cu03lbl' in name: continue
    # if not 'multis' in name: continue
    if not 'cu01' in name and not 'cuhk01' in name: continue
    # path = paths[0]
    if not osp.exists(path + '/conf.pkl'):
        continue
    args = pickle_load(path + '/conf.pkl')
    # if osp.exists(args.logs_dir + '/eval/res.json'):
    res_path = args.logs_dir + '/eval/res.json'
    # else:
    #     res_path = args.logs_dir + '/res.json'

    if not osp.exists(res_path):
        continue
    # print(args.logs_dir)
    res = json_load(res_path)
    # print(args, res)
    res_dict[name] = res
    cfg_dict[name] = args

df = pd.DataFrame(res_dict).T
df *= 100


def f1(x):
    return r'%.2f' % x


print(df[['top-1',
          # 'mAP',
          'top-5', 'top-10',
          ]].to_latex(formatters=[f1, ] * 3))
print(df[['top-1.rk',
          # 'mAP.rk',
          'top-5.rk','top-10.rk',
          ]].to_latex(formatters=[f1, ] * 3))


