from scipy.spatial.distance import cdist

import lz
from lz import *
import cv2

feas_dict = msgpack_load(work_path + '/reid.person/fea.mp', allow_np=True)

feask = list(feas_dict.keys())[:]
feas = list(feas_dict.values())[:]
# feask = list(feas_dict.keys())
# feas = list(feas_dict.values())
feas = np.asarray(feas)
print(feas.shape)
feas_norm = np.linalg.norm(feas, ord=2, axis=1, keepdims=True)
feas_normalized = feas / feas_norm

# dist = cdist(feas_normalized, feas_normalized)
# dist[np.arange(569), np.arange(569)] = np.nan
# plt.matshow(dist)
# plt.colorbar()
# plt.show()

from sklearn.manifold import TSNE

embed2 = TSNE(n_components=2).fit_transform(
    feas_normalized
)
height = 128
width = 64
embed2 -= embed2.min(axis=0)
# np.median(np.abs(np.diff(embed2, axis=0)), axis=0)
embed2 *= width
embed2 = embed2.astype(int)
extend = np.array([height, width, ])
# embed2 += extend

shape = tuple(
    (embed2.max(axis=0).astype(int) + extend
     ).tolist()) + (3,)
print(shape)
res = np.ones(
    shape
).astype(np.uint8) * 255

for ind in range(feas.shape[0]):
    img_name = feask[ind]
    img = cv2.imread(img_name)
    img = cv2.resize(img, (width, height))
    embed = embed2[ind].astype(int)
    row, col = embed
    h, w, _ = img.shape
    # tmp = res[row:row + h, col:col + w]
    res[row:row + h, col:col + w] = img
    # if ind > 10:
    #     break
# cv2.namedWindow('tmp', cv2.WINDOW_NORMAL)
# cv2.imshow('tmp', res)
# cv2.waitKey(0)
cv2.imwrite(work_path + '/tmp3.jpg', res)
