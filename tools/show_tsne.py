from lz import *
import cv2
from sklearn.manifold import TSNE

feas_dict = msgpack_load(work_path + 'tmp.mp', allow_np=True)

limit_size = 5000
feask = list(feas_dict.keys())[:limit_size]
feas = list(feas_dict.values())[:limit_size]
feas = np.asarray(feas)
feas_norm = np.linalg.norm(feas, ord=2, axis=1, keepdims=True)
feas_normalized = feas / feas_norm

# dist = cdist(feas_normalized, feas_normalized)
# dist[np.arange(569), np.arange(569)] = np.nan
# plt.matshow(dist)
# plt.colorbar()
# plt.show()

embed2 = TSNE(n_components=2).fit_transform(
    feas_normalized
)

height = 128
width = 64
embed2 = embed2 - embed2.min(axis=0)
# np.median(np.abs(np.diff(embed2, axis=0)), axis=0)
space = 64+16
embed2 *= space
embed2 = embed2.astype(int)
extend = np.array([height, width, ])

shape = tuple(
    (embed2.max(axis=0).astype(int) + extend).tolist()
) + (3,)
print('res shape', shape)
res = np.ones(
    shape
).astype(np.uint8) * 255

for ind in range(feas.shape[0]):
    img_name = feask[ind]
    img = cv2.imread(img_name)
    img2 = cvb.resize_keep_ar(img, height, width)
    if not (img2.shape[0] <= height and img2.shape[1] <= width):
        img2 = cvb.resize_keep_ar(img, width, width)
    assert (img2.shape[0] <= height and img2.shape[1] <= width)
    # if img2.shape[0] < height:
    #     img2 = np.concatenate((img2, np.ones((height - img2.shape[0], img2.shape[1], 3)) * 255), axis=0)
    # if img2.shape[1] < width:
    #     img2 = np.concatenate((img2, np.ones((img2.shape[0], width - img2.shape[1], 3)) * 255), axis=1)
    # assert img2.shape[0] == height and img2.shape[1] == width
    img = img2
    embed = embed2[ind].astype(int)
    row, col = embed
    h, w, _ = img.shape
    res[row:row + h, col:col + w] = img
    # if ind > 10:
    #     break
# cv2.namedWindow('tmp', cv2.WINDOW_NORMAL)
# cv2.imshow('tmp', res)
# cv2.waitKey(0)
cv2.imwrite(work_path + '/tmp.jpg', res)
