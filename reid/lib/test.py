from lz import *
import lz
from reid.lib.cython_eval import eval_market1501_wrap

# num_q = 1980
# num_g = 9330
#
# distmat = np.random.rand(num_q, num_g).astype(np.float32) * 20
# q_pids = np.random.randint(0, min(num_q, num_g), size=num_q, dtype=np.int64)
# g_pids = np.random.randint(0, min(num_q, num_g), size=num_g, dtype=np.int64)
# q_camids = np.random.randint(0, 5, size=num_q, dtype=np.int64)
# g_camids = np.random.randint(0, 5, size=num_g, dtype=np.int64)

distmat, q_pids, g_pids, q_camids, g_camids = lz.msgpack_load(work_path + 'tmp.mp')
# distmat, q_pids, g_pids, q_camids, g_camids = list(map(np.array, [distmat, q_pids, g_pids, q_camids, g_camids]))
num_q = q_pids.shape[0]
num_g = g_pids.shape[0]
# for mat in [distmat, q_pids, g_pids, q_camids, g_camids]:
#     print(mat.dtype)

tic = time.time()
mAP, cmc = eval_market1501_wrap(distmat,
                                q_pids,
                                g_pids,
                                q_camids,
                                g_camids, 10)
toc = time.time()
print('\nconsume time {} \n mAP is {} \n cmc is {}\n'.format(toc - tic, mAP, cmc))


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    if distmat.shape[0] < 65535:
        indices = indices.astype(np.uint16)
    else:
        indices = indices.astype(np.uint32)
    matches = (g_pids[indices] == q_pids[:, np.newaxis])

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # keep = np.invert(remove)
        keep = (g_pids[order] != q_pid) | (g_camids[order] != q_camid)
        # compute cmc curve
        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        if num_rel == 0:
            AP = 0
        else:
            AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return mAP, all_cmc

# tic = time.time()
# mAP, cmc = eval_market1501(distmat,
#                            q_pids,
#                            g_pids,
#                            q_camids,
#                            g_camids, 10)
# toc = time.time()
# print('\nconsume time {} \n mAP is {} \n cmc is {}\n'.format(toc - tic, mAP, cmc))

# from reid.lib.cython_eval import my_cusum, my_sum
# t = np.ones(5)
# tt = np.empty(5)
# print(tt)
# my_cusum(t, tt, 4)
# print(tt)
# print(my_sum(t, 4))
