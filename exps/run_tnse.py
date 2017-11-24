from sklearn import manifold, datasets
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from matplotlib.patches import Ellipse
from lz import *

from sne.wrapper import Wrapper
# from tsne import TSNE
from sne.vtsne import VTSNE
import lz
from reid import datasets

lz.init_dev((0,))

path = 'ohmn_match/2'

def preprocess(perplexity=30, metric='euclidean'):
    """ Compute pairiwse probabilities for MNIST pixels.
    """
    db = lz.Database('distmat.h5', 'a')
    print(db.keys())
    distmat = db[path]
    ds = datasets.create('cuhk03', '/home/xinglu/.torch/data/cuhk03/', split_id=0)
    y = np.asarray([
        pid for fn, pid, cid in ds.query
    ])

    n_points = distmat.shape[0]
    distances2 = distmat
    # This return a n x (n-1) prob array
    pij = manifold.t_sne._joint_probabilities(distances2, perplexity, False)
    # Convert to n x n prob array
    pij = squareform(pij)
    return n_points, pij, y


n_points, pij2d, y = preprocess()

draw_ellipse = True
i, j = np.indices(pij2d.shape)
i = i.ravel()
j = j.ravel()
pij = pij2d.ravel().astype('float32')
# Remove self-indices
idx = i != j
i, j, pij = i[idx], j[idx], pij[idx]

n_topics = 2
n_dim = 2
print(n_points, n_dim, n_topics)

lz.mkdir_p('work/'+path)
os.chdir('work/'+path)
model = VTSNE(n_points, n_topics, n_dim)
wrap = Wrapper(model, batchsize=4096, epochs=1)
for itr in range(250):
    print(itr, end='  ')
    wrap.fit(pij, i, j)

    # Visualize the results
    embed = model.logits.weight.cpu().data.numpy()
    f = plt.figure()
    if not draw_ellipse:
        plt.scatter(embed[:, 0], embed[:, 1], c=y * 1.0 / y.max())
        plt.axis('off')
        plt.savefig('scatter_{:03d}.png'.format(itr), bbox_inches='tight')
        plt.close(f)
    else:
        # Visualize with ellipses
        var = np.sqrt(model.logits_lv.weight.clone().exp_().cpu().data.numpy())
        ax = plt.gca()
        for xy, (w, h), c in zip(embed, var, y):
            e = Ellipse(xy=xy, width=w, height=h, ec=None, lw=0.0)
            e.set_facecolor(plt.cm.Paired(c * 1.0 / y.max()))
            e.set_alpha(0.5)
            ax.add_artist(e)
        ax.set_xlim(-9, 9)
        ax.set_ylim(-9, 9)
        plt.axis('off')
        plt.savefig('scatter_{:03d}.png'.format(itr), bbox_inches='tight')
        plt.close(f)
shell('convert -delay 10 -loop 0 scatter_*.png /home/xinglu/work/{}.gif'.format(path))
os.chdir('../..')