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
    # path = paths[0]
    if not osp.exists(path + '/conf.pkl'):
        continue
    args = pickle_load(path + '/conf.pkl')
    if not osp.exists(args.logs_dir + '/res.json'):
        continue
    # print(args.logs_dir)
    res = json_load(args.logs_dir + '/res.json')
    # print(args, res)
    res_dict[name] = res
    cfg_dict[name] = args

df = pd.DataFrame(res_dict).T
df *= 100


def f1(x):
    return r'%.2f' % x


print(df[['top-1', 'top-5', 'top-10', ]].to_latex(formatters=[f1, ] * 3))
print(df[['top-1.rk', 'top-5.rk', 'top-10.rk', ]].to_latex(formatters=[f1, ] * 3))
# 'top-1.rk', 'top-5', 'top-5.rk',
