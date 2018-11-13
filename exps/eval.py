import sys

sys.path.insert(0, '/data1/xinglu/prj/open-reid')

from lz import *
import lz
from reid.lib import eval_market1501_wrap

features, query, gallery = msgpack_load(work_path + 'large.pk')
print(len(features), len(gallery), len(query))
lz.timer.since_last_check(f'start')
query_ids = [pid for _, pid, _ in query]
gallery_ids = [pid for _, pid, _ in gallery]
query_cams = [cam for _, _, cam in query]
gallery_cams = [cam for _, _, cam in gallery]

query_ids = np.asarray(query_ids)
gallery_ids = np.asarray(gallery_ids)
query_cams = np.asarray(query_cams)
gallery_cams = np.asarray(gallery_cams)

xx = np.vstack([features[f] for f, _, _ in query])
yy = np.vstack([features[f] for f, _, _ in gallery])
import faiss

index = faiss.IndexFlatL2(xx.shape[1])
index.add(yy)
_, ranklist = index.search(xx, yy.shape[0])
mAP, all_cmc, all_AP = eval_market1501_wrap(
    np.asarray(ranklist), query_ids, gallery_ids, query_cams,
    gallery_cams,
    max_rank=10, faiss=True)
lz.timer.since_last_check(f'faiss {mAP} {all_cmc}')
