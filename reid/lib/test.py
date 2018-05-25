from reid.lib.cython_eval import bbox_overlaps, eval_market1501_wrap
import lz
from lz import *
from reid.evaluators import eval_market1501

# print(np.random.rand(5, 4).dtype)
# res = bbox_overlaps(np.random.rand(10, 4), np.random.rand(5, 4))
# res = np.asarray(res)
# print(res)

# with lz.Database(lz.root_path + '/exps/work/eval/eval.h5', 'r') as db:
#     print(list(db.keys()))
#     for name in ['distmat', 'query_ids', 'gallery_ids', 'query_cams', 'gallery_cams']:
#         locals()[name] = db['test/' + name]
#     timer = lz.Timer()
#     # q_pids, g_pids, q_camids, g_camids = query_ids, gallery_ids, query_cams, gallery_cams
#     # res = eval_market1501_wrap(distmat, query_ids, gallery_ids, query_cams, gallery_cams, 10)
#     # print(res, distmat.shape)
#     timer.since_start('consume ')

# np.random.seed(16)
num_q = 15
num_g = 150

distmat = np.random.rand(num_q, num_g).astype(np.float32) * 20
q_pids = np.random.randint(0, min(num_q, num_g), size=num_q, dtype=np.int64)
g_pids = np.random.randint(0, min(num_q, num_g), size=num_g, dtype=np.int64)
q_camids = np.random.randint(0, 5, size=num_q, dtype=np.int64)
g_camids = np.random.randint(0, 5, size=num_g, dtype=np.int64)
tic = time.time()
mAP, cmc = eval_market1501_wrap(distmat,
                                q_pids,
                                g_pids,
                                q_camids,
                                g_camids, 10)
toc = time.time()
print('\nconsume time {} \n mAP is {} \n cmc is {}\n'.format(toc - tic, mAP, cmc))

tic = time.time()
mAP, cmc = eval_market1501(distmat,
                           q_pids,
                           g_pids,
                           q_camids,
                           g_camids, 10)
toc = time.time()
print('\nconsume time {} \n mAP is {} \n cmc is {}\n'.format(toc - tic, mAP, cmc))

# from reid.lib.cython_eval import my_cusum, my_sum
# t = np.ones(5)
# tt = np.empty(5)
# print(tt)
# my_cusum(t, tt, 4)
# print(tt)
# print(my_sum(t, 4))
