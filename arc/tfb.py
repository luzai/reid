from lz import *
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator


class Loader(object):
    def __init__(self, name, path):
        self.name = name
        if 'events.out.tfevents' not in path:
            paths = glob.glob(path + '/events.out.tfevents*')
            assert len(paths) == 1
            path = paths[0]
        self.path = path
        self.em = event_accumulator.EventAccumulator(
            size_guidance={event_accumulator.COMPRESSED_HISTOGRAMS: 1,
                           event_accumulator.IMAGES: 1,
                           event_accumulator.AUDIO: 1,
                           event_accumulator.SCALARS: 0,
                           event_accumulator.HISTOGRAMS: 1,
                           event_accumulator.TENSORS: 0},
            path=path)

        self.reload()

    def reload(self):
        tic = time.time()
        self.em.Reload()
        # logger.info('reload consume time {}'.format(time.time() - tic))
        self.scalars_names = self.em.Tags()['scalars']
        # self.tensors_names = self.em.Tags()['tensors']


class ScalarLoader(Loader):
    def __init__(self, path, name=None, ):
        if name is None:
            name = path.split('/')[-1]
        super(ScalarLoader, self).__init__(name, path)

    def load_scalars(self, reload=False):
        if reload:
            self.reload()
        scalars_df = pd.DataFrame()
        for scalar_name in self.scalars_names:
            if 'test' not in scalar_name: continue
            for e in self.em.Scalars(scalar_name):
                iter = e.step
                val = e.value
                val *= 100
                # from decimal import Decimal
                # val = Decimal(val)
                val = round(val, 2)
                scalars_df.loc[iter, scalar_name.split('/')[-1]] = val

        return scalars_df.sort_index().sort_index(axis=1)


dfs = []
names = []
# for path in ['work/cu03det.search.1e-01.0e+00.run0',
#              'work/cu03det.search.1e-01.1e-03.run0',
#              'work/cu03det.cent.dis.1e-03',
#              'work/cu03det.cent.dis.dop.0.33.run2', ]:
for path in glob.glob('work/tri.dep*'):
    assert osp.exists(path)
    df = ScalarLoader(path=path).load_scalars()
    if df.index.max() != 66: continue
    df = df.max()
    # df = df.iloc[-1, :]
    name = path.split('/')[-1]
    names.append(name)
    dfs.append(df.to_frame(name=name))

df = pd.concat(dfs, axis=1)
df = df.transpose()
df = df[['top-1', 'top-5', 'top-10']]
# df3 = df[['top-1.rk', 'top-5.rk', 'mAP.rk']]
# print(df)
print(df2md(df))
print(df.to_latex())
