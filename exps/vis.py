import sys

sys.path.insert(0, '/data1/xinglu/prj/open-reid')

from lz import *
import lz

np.random.seed(16)
random.seed(16)
[distmat, xx, yy, query, gallery, all_AP] = msgpack_load(work_path + 't.pk')
query_ps = [path for path, _, _ in query]
gallery_ps = [path for path, _, _ in gallery]
query_ids = [pid for _, pid, _ in query]
gallery_ids = [pid for _, pid, _ in gallery]
query_cams = [cam for _, _, cam in query]
gallery_cams = [cam for _, _, cam in gallery]

query_ids = np.asarray(query_ids)
gallery_ids = np.asarray(gallery_ids)
query_cams = np.asarray(query_cams)
gallery_cams = np.asarray(gallery_cams)


def plot_pairs(k, topk=3, allk=5, figsize=None):  # (12,6)
    query_id = query_ids[k]
    query_cam = query_cams[k]
    qpath = query_ps[k]
    fig, axes = plt.subplots(nrows=1, ncols=allk + 1,
                             figsize=figsize
                             )
    axes = np.ravel(axes)
    q = (cvb.read_img(qpath))[..., ::-1]
    plt_imshow(q, axes[0])
    axes[0].set_title(f'query_id {query_id}')

    axes_top = axes[1:topk + 1]
    axes_correct = axes[topk + 1:]

    gidxs = []  # pred  randlist
    gts = []
    for gidx in np.argsort(distmat[k]):
        if gallery_ids[gidx] != query_id or gallery_cams[gidx] != query_cam:
            gidxs.append(gidx)
            if gallery_ids[gidx] == query_id:
                gts.append(True)
            else:
                gts.append(False)
    gidxs_correct = ([gidxs[t] for t in np.where(gts)[0]])
    gidxs_correct_ind = np.where(gts)[0]
    #     print( gidxs[:5],'\n', gidxs_correct )
    for ind, (ax, gidx, gt) in enumerate(zip(axes_top, gidxs, gts)):
        gpath = gallery_ps[gidx]
        g = cvb.read_img(gpath)[..., ::-1]
        ax.set_title(f'rank {ind}')
        if gt:
            c = 'green'
        else:
            c = 'red'
        plt_imshow_board(g, ax, c)

    for ind, (ax, gidx, gidx_ind) in enumerate(zip(axes_correct, gidxs_correct, gidxs_correct_ind)):
        gpath = gallery_ps[gidx]
        g = cvb.read_img(gpath)[..., ::-1]
        ax.set_title(f'rank {gidx_ind+topk}')
        plt_imshow_board(g, ax, 'green')
    if ind < allk - topk:
        for ind2 in range(ind + 1, allk - topk):
            plt_imshow(np.ones((256, 128, 3)) * 255, axes_correct[ind2], )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0)
    # plt.savefig(work_path + f'reid.vis/{randomword()}.png')
    plt.savefig(work_path + f'reid.vis/{k}.png')
    return True


mkdir_p(work_path + 'reid.vis')
# hardlist = [2267, 3188, 753, 2059, 2025, 1058, 2393,
#             1280, 2650, 1142, 2795, 476
#             ]
hardlist = [2267, 3188, 753,
            2795, 476, 1142,
            ]
topk = 13
topk = 5
for qidx in hardlist:
    _ = plot_pairs(qidx, 3, topk, (topk * 128 // 100 + 1, 256 // 100 + 1))  # 13*128 //100

imgl = []
negpadding = 25
for f in glob.glob(work_path + 'reid.vis/*.png'):
    img = cvb.read_img(f)
    img = img[negpadding:-negpadding, :, :]
    imgl.append(img)
img = np.vstack(imgl)
cvb.write_img(img, work_path + 'final.png')
