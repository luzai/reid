import pyximport
pyximport.install()

from reid.lib.cython_eval import bbox_overlaps, eval_market1501_wrap
import lz
from lz import *

# print(np.random.rand(5, 4).dtype)
# res = bbox_overlaps(np.random.rand(10, 4), np.random.rand(5, 4))
# res = np.asarray(res)
# print(res)

with lz.Database(lz.root_path + '/exps/work/eval/eval.h5', 'r') as db:
    print(list(db.keys()))
    for name in ['distmat', 'query_ids', 'gallery_ids', 'query_cams', 'gallery_cams']:
        locals()[name] = db['test/' + name]
    timer = lz.Timer()
    res = eval_market1501_wrap(distmat, query_ids, gallery_ids, query_cams, gallery_cams, 10)
    print(res)
    timer.since_start()

# (0.72182631341475978, [ 0.89459622  0.9287411   0.9438836   0.95457244  0.96080762  0.96466744
#   0.96793348  0.97030878  0.97209024  0.97298098]
#  dtype:float32 shape:(10,))
# 291.666

